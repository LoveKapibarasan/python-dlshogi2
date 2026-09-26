// Native UCT search core for python-dlshogi2.
//
// This is a C++ port of MCTSPlayer.search / uct_search / select_max_ucb_child /
// eval_node from pydlshogi2/player/mcts_player.py.  The tree, the selection
// formula, virtual loss, batching, the discard rule, mate / repetition /
// nyugyoku detection and the backup are the same algorithm, written so that
// the floating point operations happen in the same precision and order as the
// Python version (float32 where it used numpy float32 arrays, double where it
// used Python floats).  The neural network stays in Python: the search fills a
// feature buffer and calls back into Python to evaluate a batch.
//
// The Python side talks to this through ctypes, so the library needs no Python
// headers.  Board handling comes from cshogi's own C++ sources, so move
// encodings and legal move order are identical to the Python engine's.
//
// Build: pydlshogi2/uct/native/build.sh

#include <cmath>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>

#include "cshogi.h"

namespace {

constexpr int MOVE_LABELS_NUM = 27 * 81;
constexpr int FEATURES_NUM = 104;  // 28 piece planes + 2 * 38 hand planes
constexpr int MAX_PIECES_IN_HAND[7] = {18, 4, 4, 4, 4, 2, 2};  // cshogi.MAX_PIECES_IN_HAND

// ノードの値。NN の評価値 [0,1] 以外に、終局を表す特別な値を持つ
constexpr float VALUE_WIN = 10000.0f;
constexpr float VALUE_LOSE = -10000.0f;
constexpr float VALUE_DRAW = 20000.0f;
constexpr float VALUE_NONE = std::numeric_limits<float>::quiet_NaN();  // 評価待ち

constexpr int VIRTUAL_LOSS = 1;
constexpr double QUEUING = -1.0;
constexpr double DISCARDED = -2.0;

// cshogi.dlshogi のラベル -> このリポジトリのラベル (打ち駒の並びだけが違う)
int LABEL_PERMUTATION[MOVE_LABELS_NUM];

void init_label_permutation() {
    for (int i = 0; i < MOVE_LABELS_NUM; i++) LABEL_PERMUTATION[i] = i;
    // dlshogi の打ち駒の並び P L N S B R G を、HAND_PIECES の P L N S G B R の添字へ
    const int dlshogi_order[7] = {0, 1, 2, 3, 5, 6, 4};
    for (int c = 0; c < 7; c++)
        for (int sq = 0; sq < 81; sq++)
            LABEL_PERMUTATION[(20 + c) * 81 + sq] = (20 + dlshogi_order[c]) * 81 + sq;
}

struct Node {
    int move_count = 0;
    double sum_value = 0.0;           // Python float
    float value = VALUE_NONE;         // NN の価値、または VALUE_WIN/LOSE/DRAW
    bool expanded = false;            // child_move が作られたか
    bool evaluated = false;           // policy がセットされたか

    std::vector<int> child_move;
    std::vector<int> child_move_count;
    std::vector<float> child_sum_value;
    std::vector<std::unique_ptr<Node>> child_node;
    std::vector<float> policy;
    std::vector<double> policy_list;  // Python の policy.tolist()
    // select が読むキャッシュ (uct_node.UctNode と同じ)
    std::vector<float> child_q;
    std::vector<float> child_unvisited;
    std::vector<float> child_policy_denom;
    std::vector<uint8_t> visited_flags;
    double visited_policy_sum = 0.0;

    void expand(const __Board& board) {
        child_move.clear();
        for (__LegalMoveList ml(board); !ml.end(); ml.next())
            child_move.push_back(ml.move());
        const size_t n = child_move.size();
        child_move_count.assign(n, 0);
        child_sum_value.assign(n, 0.0f);
        child_node.clear();
        child_node.resize(n);
        child_q.assign(n, 0.0f);
        child_unvisited.assign(n, 1.0f);
        child_policy_denom.assign(n, 0.0f);
        visited_flags.assign(n, 0);
        visited_policy_sum = 0.0;
        policy.clear();
        policy_list.clear();
        expanded = true;
    }

    // UctNode.set_policy
    void set_policy(const std::vector<float>& p) {
        policy = p;
        const size_t n = p.size();
        policy_list.resize(n);
        for (size_t i = 0; i < n; i++) policy_list[i] = (double)p[i];
        float visited_sum = 0.0f;  // numpy float32 sum (訪問済みの子は評価時点では普通 0 個)
        for (size_t i = 0; i < n; i++) {
            const int count = child_move_count[i];
            child_policy_denom[i] = p[i] / (float)(count + 1);
            const bool visited = count != 0;
            visited_flags[i] = visited ? 1 : 0;
            child_unvisited[i] = visited ? 0.0f : 1.0f;
            if (visited) visited_sum += p[i];
        }
        visited_policy_sum = (double)visited_sum;
        evaluated = true;
    }

    // UctNode.refresh_child
    void refresh_child(int i) {
        const int count = child_move_count[i];
        const double prior = policy_list[i];
        if (count) {
            if (!visited_flags[i]) {
                visited_flags[i] = 1;
                child_unvisited[i] = 0.0f;
                visited_policy_sum += prior;
            }
            child_q[i] = (float)((double)child_sum_value[i] / count);
        } else {
            if (visited_flags[i]) {
                visited_flags[i] = 0;
                child_unvisited[i] = 1.0f;
                visited_policy_sum -= prior;
            }
            child_q[i] = 0.0f;
        }
        child_policy_denom[i] = (float)(prior / (1 + count));
    }
};

bool is_terminal_value(float v) {
    return v == VALUE_WIN || v == VALUE_LOSE || v == VALUE_DRAW;
}

struct QueueEntry {
    Node* node;
    int color;
};

typedef int (*EvalCallback)(int n);
typedef int (*InterruptCallback)(void);
typedef int (*LaunchCallback)(int slot, int n);
typedef int (*WaitCallback)(int slot);

typedef std::vector<std::pair<Node*, int>> Trajectory;

// 1 バッチ分の入出力と、その評価を待っている経路
struct Slot {
    float* features;      // [batch][104][81]
    float* policy_out;    // [batch][2187]
    float* value_out;     // [batch]
    std::vector<QueueEntry> queue;
    int count = 0;                    // キューに積んだ局面の数
    std::vector<Trajectory> batch;    // 評価待ちの経路
    int nbatch = 0;
    std::vector<Trajectory> discarded;
};

struct Searcher {
    int batch_size;
    Slot slots[2];
    Slot* cur;                        // queue_node が積む先
    EvalCallback eval_cb = nullptr;
    InterruptCallback interrupt_cb = nullptr;
    // 設定されていれば、GPU が 1 バッチを評価している間に次のバッチを集める
    LaunchCallback launch_cb = nullptr;
    WaitCallback wait_cb = nullptr;

    double c_puct = 1.0;
    double c_base = 19652.0;
    double fpu_reduction = 0.27;
    float temperature = 1.0f;
    int mate_tree_ply = 3;

    __Board root_board;
    std::unique_ptr<Node> game_root;  // 開始局面のノード
    Node* current_head = nullptr;
    std::string start_position;       // "startpos" / "sfen ..."
    std::vector<int> moves;           // 開始局面からの指し手

    long long playout_count = 0;

    // バッファは nslots 個のスロットが連続して並んでいる
    Searcher(int bs, int nslots, float* f, float* p, float* v) : batch_size(bs) {
        for (int k = 0; k < 2; k++) {
            const int o = k < nslots ? k : 0;
            slots[k].features = f + (size_t)o * bs * FEATURES_NUM * 81;
            slots[k].policy_out = p + (size_t)o * bs * MOVE_LABELS_NUM;
            slots[k].value_out = v + (size_t)o * bs;
            slots[k].queue.resize(bs);
            slots[k].batch.resize(bs);
        }
        cur = &slots[0];
        game_root.reset(new Node());
        current_head = game_root.get();
    }

    // ---- 入力特徴量 (features.make_input_features) ----
    void make_input_features(const __Board& board, float* f) {
        std::memset(f, 0, sizeof(float) * FEATURES_NUM * 81);
        const int turn = board.turn();
        if (turn == Black) board.piece_planes((char*)f);
        else board.piece_planes_rotate((char*)f);
        int i = 28;
        const int first = turn, second = 1 - turn;
        for (const int color : {first, second}) {
            const std::vector<int> hand = board.pieces_in_hand(color);
            for (int k = 0; k < 7; k++) {
                const int num = hand[k];
                for (int j = 0; j < num; j++)
                    std::fill(f + (i + j) * 81, f + (i + j + 1) * 81, 1.0f);
                i += MAX_PIECES_IN_HAND[k];
            }
        }
    }

    void queue_node(const __Board& board, Node* node) {
        make_input_features(board, cur->features + (size_t)cur->count * FEATURES_NUM * 81);
        cur->queue[cur->count] = {node, board.turn()};
        cur->count++;
    }

    // ---- 評価 (MCTSPlayer.eval_node) ----
    void eval_node() {
        eval_cb(slots[0].count);
        apply_eval(slots[0]);
    }

    // 評価結果をノードに書き込む
    void apply_eval(Slot& slot) {
        const int n = slot.count;
        std::vector<float> logits;
        for (int i = 0; i < n; i++) {
            Node* node = slot.queue[i].node;
            const int color = slot.queue[i].color;
            const float* pl = slot.policy_out + (size_t)i * MOVE_LABELS_NUM;
            const size_t m = node->child_move.size();
            logits.resize(m);
            for (size_t j = 0; j < m; j++)
                logits[j] = pl[LABEL_PERMUTATION[__dlshogi_make_move_label(node->child_move[j], color)]];
            // softmax_temperature_with_normalize (float32)
            float maxv = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < m; j++) {
                logits[j] /= temperature;
                if (logits[j] > maxv) maxv = logits[j];
            }
            float sum = 0.0f;
            for (size_t j = 0; j < m; j++) {
                logits[j] = std::exp(logits[j] - maxv);
                sum += logits[j];
            }
            for (size_t j = 0; j < m; j++) logits[j] /= sum;
            node->set_policy(logits);
            node->value = slot.value_out[i];
        }
    }

    // ---- 選択 (select_max_ucb_child) ----
    int select_max_ucb_child(Node* node) {
        const size_t n = node->child_move.size();
        if (node->move_count == 0) {
            // policy.argmax()
            int best = 0;
            for (size_t i = 1; i < n; i++)
                if (node->policy[i] > node->policy[best]) best = (int)i;
            return best;
        }
        const float fpu = (float)(node->sum_value / node->move_count
                                  - fpu_reduction * std::sqrt(node->visited_policy_sum));
        const double c = std::log((node->move_count + c_base + 1) / c_base) + c_puct;
        const float scale = (float)(c * std::sqrt((double)node->move_count));
        int best = 0;
        float best_ucb = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < n; i++) {
            // ucb = unvisited*fpu + q + denom*scale (numpy と同じ順に float32 で)
            float ucb = node->child_unvisited[i] * fpu;
            ucb += node->child_q[i];
            const float u = node->child_policy_denom[i] * scale;
            ucb += u;
            if (ucb > best_ucb) { best_ucb = ucb; best = (int)i; }
        }
        return best;
    }

    static void update_result(Node* node, int idx, double result) {
        node->sum_value += result;
        node->move_count += 1 - VIRTUAL_LOSS;
        node->child_sum_value[idx] += (float)result;
        node->child_move_count[idx] += 1 - VIRTUAL_LOSS;
        node->refresh_child(idx);
    }

    // ---- 1 回の降下 (uct_search) ----
    double uct_search(__Board& board, Node* node, std::vector<std::pair<Node*, int>>& traj) {
        const int idx = select_max_ucb_child(node);
        board.push(node->child_move[idx]);
        node->move_count += VIRTUAL_LOSS;
        node->child_move_count[idx] += VIRTUAL_LOSS;
        node->refresh_child(idx);
        traj.emplace_back(node, idx);

        double result;
        if (!node->child_node[idx]) {
            node->child_node[idx].reset(new Node());
            Node* child = node->child_node[idx].get();
            const int draw = board.isDraw(std::numeric_limits<int>::max());
            if (draw != NotRepetition) {
                if (draw == RepetitionDraw) {
                    child->value = VALUE_DRAW;
                    result = 0.5;
                } else if (draw == RepetitionWin || draw == RepetitionSuperior) {
                    child->value = VALUE_WIN;
                    result = 0.0;
                } else {
                    child->value = VALUE_LOSE;
                    result = 1.0;
                }
            } else if (board.is_nyugyoku() || board.mateMove(mate_tree_ply)) {
                child->value = VALUE_WIN;
                result = 0.0;
            } else {
                child->expand(board);
                if (child->child_move.empty()) {
                    child->value = VALUE_LOSE;
                    result = 1.0;
                } else {
                    queue_node(board, child);
                    return QUEUING;
                }
            }
        } else {
            Node* next = node->child_node[idx].get();
            if (std::isnan(next->value)) return DISCARDED;
            if (next->value == VALUE_WIN) result = 0.0;
            else if (next->value == VALUE_LOSE) result = 1.0;
            else if (next->value == VALUE_DRAW) result = 0.5;
            else if (next->child_move.empty()) result = 1.0;
            else result = uct_search(board, next, traj);
        }
        if (result == QUEUING || result == DISCARDED) return result;
        update_result(node, idx, result);
        return 1.0 - result;
    }

    // 盤を root に戻す。uct_search は経路 1 段ごとにちょうど 1 手 push する
    void unwind(size_t pushed) {
        for (size_t i = 0; i < pushed; i++) root_board.pop();
    }

    // ---- 探索ループ (search) ----
    // 1 バッチ分の降下を行い、評価待ちの経路を slot に集める
    void collect(Slot& slot) {
        cur = &slot;
        slot.count = 0;
        slot.nbatch = 0;
        slot.discarded.clear();
        for (int i = 0; i < batch_size; i++) {
            auto& traj = slot.batch[slot.nbatch];
            traj.clear();
            const double result = uct_search(root_board, current_head, traj);
            unwind(traj.size());
            if (result != DISCARDED) {
                playout_count++;
            } else {
                slot.discarded.push_back(traj);
                if ((int)slot.discarded.size() > batch_size / 2) break;
            }
            if (result == QUEUING) slot.nbatch++;  // 評価待ちの経路だけ残す
        }
    }

    // 評価済みの slot について、破棄した経路の Virtual Loss を戻し、結果をバックアップする
    void finish(Slot& slot) {
        for (auto& traj : slot.discarded)
            for (auto& [node, idx] : traj) {
                node->move_count -= VIRTUAL_LOSS;
                node->child_move_count[idx] -= VIRTUAL_LOSS;
                node->refresh_child(idx);
            }
        for (int b = 0; b < slot.nbatch; b++) {
            auto& traj = slot.batch[b];
            double result = 0.0;
            bool leaf = true;
            for (auto it = traj.rbegin(); it != traj.rend(); ++it) {
                Node* node = it->first;
                const int idx = it->second;
                if (leaf) {
                    result = 1.0 - (double)node->child_node[idx]->value;
                    leaf = false;
                }
                update_result(node, idx, result);
                result = 1.0 - result;
            }
        }
    }

    bool should_stop(long long max_playouts) {
        if (max_playouts >= 0 && playout_count >= max_playouts) return true;
        return interrupt_cb && interrupt_cb();
    }

    long long search(long long max_playouts) {
        if (!launch_cb) {
            // MCTSPlayer.search と同じ逐次版
            while (true) {
                collect(slots[0]);
                if (slots[0].nbatch > 0) eval_node();
                finish(slots[0]);
                if (should_stop(max_playouts)) return playout_count;
            }
        }
        // 2 スロットを交互に使い、GPU の評価と次のバッチの降下を重ねる
        int a = 0;
        collect(slots[a]);
        if (slots[a].nbatch > 0) launch_cb(a, slots[a].count);
        while (true) {
            const int b = 1 - a;
            collect(slots[b]);
            if (slots[b].nbatch > 0) launch_cb(b, slots[b].count);
            if (slots[a].nbatch > 0) {
                wait_cb(a);
                apply_eval(slots[a]);
            }
            finish(slots[a]);
            if (should_stop(max_playouts)) {
                if (slots[b].nbatch > 0) {
                    wait_cb(b);
                    apply_eval(slots[b]);
                }
                finish(slots[b]);
                return playout_count;
            }
            a = b;
        }
    }

    // ---- 局面の設定と木の再利用 (NodeTree.reset_to_position の簡略版) ----
    int set_position(const std::string& start, const std::vector<int>& new_moves) {
        bool reuse = start == start_position && new_moves.size() >= moves.size()
                     && std::equal(moves.begin(), moves.end(), new_moves.begin());
        root_board.set(start == "startpos" ? std::string(DefaultStartPositionSFEN) : start);
        for (int m : new_moves) root_board.push(m);
        if (!reuse) {
            game_root.reset(new Node());
            current_head = game_root.get();
        } else {
            for (size_t k = moves.size(); k < new_moves.size(); k++)
                current_head = descend(current_head, new_moves[k]);
        }
        start_position = start;
        moves = new_moves;
        return 0;
    }

    // current_head の子のうち move を残して他を捨て、その子を返す
    Node* descend(Node* node, int move) {
        std::unique_ptr<Node> keep;
        if (node->expanded) {
            for (size_t i = 0; i < node->child_move.size(); i++)
                if (node->child_move[i] == move) {
                    keep = std::move(node->child_node[i]);
                    break;
                }
        }
        if (!keep) keep.reset(new Node());
        // 祖先はもう探索しないので、統計を捨てて子 1 つだけ持つ
        node->child_move.assign(1, move);
        node->child_node.clear();
        node->child_node.push_back(std::move(keep));
        node->child_move_count.clear();
        node->child_sum_value.clear();
        node->policy.clear();
        node->policy_list.clear();
        node->child_q.clear();
        node->child_unvisited.clear();
        node->child_policy_denom.clear();
        node->visited_flags.clear();
        node->expanded = false;
        node->evaluated = false;
        return node->child_node[0].get();
    }
};

}  // namespace

extern "C" {

// バッファは nslots (1 か 2) スロット分、スロットごとに batch_size 局面ぶん並べる
void* uct_new(int batch_size, int nslots, float* features, float* policy_out, float* value_out) {
    static bool initialized = false;
    if (!initialized) {
        // cshogi のモジュール初期化と同じ
        initTable();
        Position::initZobrist();
        HuffmanCodedPos_init();
        PackedSfen_init();
        Book_init();
        init_label_permutation();
        initialized = true;
    }
    return new Searcher(batch_size, nslots, features, policy_out, value_out);
}

void uct_free(void* h) { delete (Searcher*)h; }

void uct_set_params(void* h, double c_puct, double c_base, double fpu_reduction,
                    double temperature, int mate_tree_ply) {
    auto* s = (Searcher*)h;
    s->c_puct = c_puct;
    s->c_base = c_base;
    s->fpu_reduction = fpu_reduction;
    s->temperature = (float)temperature;
    s->mate_tree_ply = mate_tree_ply;
}

void uct_set_callbacks(void* h, EvalCallback eval_cb, InterruptCallback interrupt_cb) {
    auto* s = (Searcher*)h;
    s->eval_cb = eval_cb;
    s->interrupt_cb = interrupt_cb;
}

// launch(slot, n) は評価を GPU に投げてすぐ戻り、wait(slot) はその完了を待つ。
// 設定すると、探索は 2 スロットで GPU と CPU を重ねて回す (nslots=2 で作ること)
void uct_set_pipeline(void* h, LaunchCallback launch_cb, WaitCallback wait_cb) {
    auto* s = (Searcher*)h;
    s->launch_cb = launch_cb;
    s->wait_cb = wait_cb;
}

// start: "startpos" または sfen 文字列。moves: cshogi の指し手 (int) の配列
int uct_set_position(void* h, const char* start, const int* moves, int n) {
    auto* s = (Searcher*)h;
    return s->set_position(start, std::vector<int>(moves, moves + n));
}

// ルートを展開し、未評価なら評価する。合法手の数を返す
int uct_prepare_root(void* h) {
    auto* s = (Searcher*)h;
    Node* root = s->current_head;
    if (!root->expanded) root->expand(s->root_board);
    if (!root->evaluated && !root->child_move.empty()) {
        s->cur = &s->slots[0];
        s->slots[0].count = 0;
        s->queue_node(s->root_board, root);
        s->eval_node();
    }
    return (int)root->child_move.size();
}

// max_playouts 回 (負なら interrupt_cb が止めるまで) 探索する。今回の探索回数を返す
long long uct_search(void* h, long long max_playouts) {
    auto* s = (Searcher*)h;
    s->playout_count = 0;
    return s->search(max_playouts);
}

long long uct_playout_count(void* h) { return ((Searcher*)h)->playout_count; }

// ルートの子の統計。返り値は子の数
int uct_root_stats(void* h, int* moves, int* counts, float* sums, int maxn) {
    auto* s = (Searcher*)h;
    Node* root = s->current_head;
    const int n = (int)root->child_move.size();
    for (int i = 0; i < n && i < maxn; i++) {
        moves[i] = root->child_move[i];
        counts[i] = root->child_move_count.empty() ? 0 : root->child_move_count[i];
        sums[i] = root->child_sum_value.empty() ? 0.0f : root->child_sum_value[i];
    }
    return n;
}

// ルートの訪問数・価値合計・NN の価値
void uct_root_info(void* h, int* move_count, double* sum_value, float* value) {
    auto* s = (Searcher*)h;
    *move_count = s->current_head->move_count;
    *sum_value = s->current_head->sum_value;
    *value = s->current_head->value;
}

// 訪問数最大の手をたどった読み筋。first_index はルートで選ぶ子
int uct_pv(void* h, int first_index, int* out, int maxlen) {
    auto* s = (Searcher*)h;
    Node* node = s->current_head;
    int len = 0;
    int idx = first_index;
    while (node->expanded && len < maxlen && idx >= 0 && idx < (int)node->child_move.size()) {
        out[len++] = node->child_move[idx];
        Node* next = node->child_node[idx].get();
        if (!next || !next->expanded || next->move_count == 0 || next->child_move_count.empty()) break;
        idx = (int)(std::max_element(next->child_move_count.begin(), next->child_move_count.end())
                    - next->child_move_count.begin());
        node = next;
    }
    return len;
}

}  // extern "C"

extern "C" {
// デバッグ用: ルート局面の入力特徴量 (104*81 floats)
void uct_debug_root_features(void* h, float* out) {
    auto* s = (Searcher*)h;
    s->make_input_features(s->root_board, out);
}
// デバッグ用: ルートの方策 (子の数だけ)
int uct_debug_root_policy(void* h, float* out, int maxn) {
    auto* s = (Searcher*)h;
    const int n = (int)s->current_head->policy.size();
    for (int i = 0; i < n && i < maxn; i++) out[i] = s->current_head->policy[i];
    return n;
}
}
