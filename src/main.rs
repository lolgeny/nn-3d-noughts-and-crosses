#![feature(generic_const_exprs, portable_simd, core_intrinsics)]

use std::{simd::{f64x8, SimdFloat, SimdPartialOrd}, marker::PhantomData, io::{repeat, stdin}, intrinsics::assume, f64::consts};

use rand::{thread_rng, Rng, RngCore};
use rayon::{prelude::{IntoParallelIterator, IntoParallelRefIterator, IndexedParallelIterator, ParallelIterator}, slice::ParallelSlice};
use float_ord::FloatOrd;

trait Activation {
    fn activate(inp: f64) -> f64;
    fn derivative(inp: f64) -> f64;
}
struct ReLU;
impl Activation for ReLU {
    fn activate(inp: f64) -> f64 {
        inp.max(0.0)
    }

    fn derivative(inp: f64) -> f64 {
        if inp > 0.0 {1.0} else {0.0}
    }
}

struct Sigmoid;
impl Activation for Sigmoid {
    fn activate(inp: f64) -> f64 {
        1.0/(1.0+consts::E.powf(-inp))
    }

    fn derivative(inp: f64) -> f64 {
        let s = Self::activate(inp);
        s * (1.0 - s)
    }
}

trait PrevForwardPropogator<const N: usize>: Sync {
    fn values(&self) -> &[f64; N];
}
impl<const N: usize> PrevForwardPropogator<N> for [f64; N] {
    fn values(&self) -> &[f64; N] {
        self
    }
}

struct PropogatedLayer<const N: usize>  {
    pre_activation: [f64; N],
    activated: [f64; N]
}
impl<const N: usize> PrevForwardPropogator<N> for PropogatedLayer<N> {
    fn values(&self) -> &[f64; N] {
        &self.activated
    }
}

fn slice_to_f64x8(s: &[f64]) -> f64x8 {
    if s.len() >= 8 {f64x8::from_slice(s)}
    else {
        let mut s2 = [0.0; 8];
        for i in 0..s.len() {
            s2[i] = s[i];
        }
        f64x8::from_array(s2)
    }
}

struct Layer<const P: usize, const N: usize, A: Activation> where [(); P*N]: Sized {
    weights: [f64; P*N],
    biases: [f64; N],
    weights_update_store: [f64; P*N],
    biases_update_store: [f64; N],
    _a: PhantomData<A>
}
impl<const P: usize, const N: usize, A: Activation + Sync> Layer<P, N, A>
where [(); P*N]: Sized {
    fn random() -> Self {
        let mut rng = thread_rng();
        let mut weights = [0.0; P*N];
        let mut biases = [0.0; N];
        for x in weights.iter_mut().chain(biases.iter_mut()) {
            *x = rng.gen_range(-1.0..=1.0);
        }
        Self {weights, biases, _a: PhantomData, weights_update_store: [0.0; P*N], biases_update_store: [0.0; N]}
    }
    fn forward_propogate(&self, prev: &impl PrevForwardPropogator<P>) -> PropogatedLayer<N> {
        let prev = prev.values();
        let mut pre_activation = [0.0; N];
        for n in 0..N {
            let mut sum = self.biases[n];
            for w in 0..P {
                sum += self.weights[n*P + w] * prev.values()[w];
            }
            pre_activation[n] = sum;
        }
        let mut activated = pre_activation;
        for n in &mut activated {
            *n = A::activate(*n);
        }
        PropogatedLayer {pre_activation, activated}
    }

    fn backward_propogate_final(&mut self,
        prop: &PropogatedLayer<N>, prev: &PropogatedLayer<P>,
        expected: &[f64; N], learning_rate: f64
    ) -> ([f64; N], [f64; N]) {
        let mut errors = prop.activated;
        for (i, error) in errors.iter_mut().enumerate() {
            *error = (*error - expected[i])*(*error - expected[i]);
        }
        let mut bases = [0.0; N];
        for n in 0..N {
            let mut base = 2.0 * (prop.activated[n] - expected[n]) * A::derivative(prop.pre_activation[n]);
            bases[n] = base;
            base *= learning_rate;
            self.biases_update_store[n] -= base;
            for w in 0..P {
                self.weights_update_store[n*P + w] -= base * prev.activated[w];
            }
        }
        (bases, errors)
    }
    fn backward_propogate<const R: usize, A2: Activation + Sync>(&mut self,
        prop: &PropogatedLayer<N>, prev: &impl PrevForwardPropogator<P>,
        next: &Layer<N, R, A2>, next_base: &[f64; R], learning_rate: f64
    ) -> [f64; N]
        where [(); N*R]: Sized {
        let mut bases = [0.0; N];
        for n in 0..N {
            let mut base_sum = 0.0;
            for r in 0..R {
                let base = next_base[r] * next.weights[r*N + n] * A::derivative(prop.pre_activation[n]);
                base_sum += base;
            }
            bases[n] = base_sum;
            self.biases_update_store[n] -= base_sum * learning_rate;
            for w in 0..P {
                self.weights_update_store[n*P + w] -= (base_sum + (R as f64) * prev.values()[w]) * learning_rate; 
            }
        }
        bases
    }
    fn end_minibatch(&mut self) {
        self.weights.iter_mut().enumerate().for_each(|(i, w)| *w += self.weights_update_store[i]);
        self.biases.iter_mut().enumerate().for_each(|(i, b)| *b += self.biases_update_store[i]);
        self.weights_update_store.fill(0.0);
        self.biases_update_store.fill(0.0);
    }
    pub fn cross(a: &Layer<P, N, A>, b: &Layer<P, N, A>) -> Layer<P, N, A> {
        let mut weights = [0.0; P*N];
        let mut random = thread_rng();
        for w in 0..P*N {
            weights[w] = match random.gen_range(0..3) {
                0 => a.weights[w],
                1 => b.weights[w],
                2 => (a.weights[w] + b.weights[w]) / 2.0,
                _ => unreachable!()
            } + if random.gen_bool(0.05) {random.gen_range(-1.0..=1.0)} else {0.0};
        }
        let mut biases = [0.0; N];
        for i in 0..N {
            biases[i] = match random.gen_range(0..3) {
                0 => a.biases[i],
                1 => b.biases[i],
                2 => (a.biases[i] + b.biases[i]) / 2.0,
                _ => unreachable!()
            } + if random.gen_bool(0.05) {random.gen_range(-1.0..=1.0)} else {0.0};
        }
        Layer { weights, biases, weights_update_store: [0.0; P*N], biases_update_store: [0.0; N], _a: PhantomData }
    }
}

fn train() {
    let mut hl1: Layer<3, 8, ReLU> = Layer::random();
    let mut hl2: Layer<8, 16, ReLU> = Layer::random();
    let mut hl3: Layer<16, 8, ReLU> = Layer::random();
    let mut out: Layer<8, 3, ReLU> = Layer::random();
    // train: 3a^2 + 2b - c
    //        sin(a)+cos(b)
    //        a/b + c
    let mut rng = thread_rng();
    let learning_rate = 0.6;
    let mut i = 0;
    let mut average_costs = [0.0; 3];
    let mut inputs = vec![[0.0; 3]; 120_000];
    for input in &mut inputs {
        input.fill_with(|| rng.gen_range(-1.0..=1.0));
    }
    let mut input_n = 0;
    loop {
        for _ in 0..128 {
            let input = inputs[input_n];
            input_n += 1;
            if input_n >= inputs.len() {input_n = 0;}
            let a = hl1.forward_propogate(&input);
            let b = hl2.forward_propogate(&a);
            let c = hl3.forward_propogate(&b);
            let d = out.forward_propogate(&c);
    
            unsafe {assume(input.len() == 3);}
            let expected_output = [
                3.0*input[0]*input[0] + 2.0*input[1] - input[2]               +20.0,
                input[0].sin() + input[1].cos()                               +20.0,
                input[0] + input[1] + input[2] * input[2]                     +20.0
            ];
    
            let (d, cost) = out.backward_propogate_final(&d, &c, &expected_output, learning_rate);
            let e = hl3.backward_propogate(&c, &b, &out, &d, learning_rate);
            let f = hl2.backward_propogate(&b, &a, &hl3, &e, learning_rate);
            let _g = hl1.backward_propogate(&a, &input, &hl2, &f, learning_rate);
            i += 1;
            for a in 0..average_costs.len() {
                average_costs[a] += cost[a]/100_000_000.0;
            }
            if i % 100_000_000 == 0{println!("{average_costs:?}"); average_costs.fill(0.0);}
        }
        hl1.end_minibatch();
        hl2.end_minibatch();
        out.end_minibatch();
    }
}


struct NACSolver {
    hl1: Layer<27, 16, ReLU>,
    hl2: Layer<16, 16, Sigmoid>,
    hl3: Layer<16, 16, ReLU>,
    out: Layer<16, 27, Sigmoid>
}
impl NACSolver {
    pub fn random() -> Self {
        let hl1 = Layer::random();
        let hl2 = Layer::random();
        let hl3 = Layer::random();
        let out = Layer::random();
        Self {hl1, hl2, hl3, out}
    }
    pub fn get_move(&self, board: &[BoardTile; 27], player_a: bool) -> usize {
        let input = board.map(|t| match t {
            BoardTile::None => 0.5,
            BoardTile::A => if player_a {0.0} else {1.0},
            BoardTile::B => if player_a {1.0} else {0.0},
        });
        let a = self.hl1.forward_propogate(&input);
        let b = self.hl2.forward_propogate(&a);
        let c = self.hl3.forward_propogate(&b);
        let out = self.out.forward_propogate(&c);
        out.activated.iter().enumerate()
        .filter(|(n, _)| board[*n] == BoardTile::None).max_by_key(|(_, k)| FloatOrd(**k)).unwrap().0
    }
    pub fn cross(a: &NACSolver, b: &NACSolver) -> NACSolver {
        NACSolver {
            hl1: Layer::cross(&a.hl1, &b.hl1),
            hl2: Layer::cross(&a.hl2, &b.hl2),
            hl3: Layer::cross(&a.hl3, &b.hl3),
            out: Layer::cross(&a.out, &b.out)
        }
    }
}

const POPULATION_SIZE: usize = 1000;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum MatchWinner {
    A, Tie, B
}
impl MatchWinner {
    pub fn announce(self) {
        match self {
            Self::A => println!("The AI won!"),
            Self::B => println!("You won!"),
            Self::Tie => println!("It's a draw!")
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum BoardTile {None, A, B}

fn has_won(board: &[BoardTile; 27]) -> Option<MatchWinner> {
    let tile_to_winner = |index| {
        match board[index] {
            BoardTile::A => MatchWinner::A,
            BoardTile::B => MatchWinner::B,
            BoardTile::None => unreachable!()
        }
    };
    if board.iter().all(|t| t != &BoardTile::None) {return Some(MatchWinner::Tie);}
    for win in [
        (0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6),
        (9, 10, 11), (12, 13, 14), (15, 16, 17), (9, 12, 15), (10, 13, 16), (11, 14, 17), (9, 13, 17), (11, 13, 15),
        (18, 19, 20), (21, 22, 23), (24, 25, 26), (18, 21, 24), (19, 22, 25), (20, 23, 26), (18, 22, 26), (20, 22, 24),
        (0, 9, 18), (1, 10, 19), (2, 11, 20), (3, 12, 21), (4, 13, 22),
            (5, 14, 23), (6, 15, 24), (7, 16, 25), (8, 17, 26),
        (0, 10, 20), (2, 10, 18), (0, 12, 24), (6, 12, 18),
        (8, 14, 20), (2, 14, 26), (8, 16, 24), (6, 16, 26),
        (3, 13, 23), (5, 13, 21), (1, 13, 25), (7, 13, 19),
        (0, 13, 26), (2, 13, 24), (6, 13, 20), (8, 13, 18)
    ] {
        if board[win.0] != BoardTile::None && board[win.0] == board[win.1] && board[win.0] == board[win.2] {
            return Some(tile_to_winner(win.0));
        }
    }
    None
}

fn fight(a: &NACSolver, b: &NACSolver) -> MatchWinner {
    let mut board = [BoardTile::None; 27];
    loop {
        let mut stalemate = true;
        let a_move = a.get_move(&board, true);
        if board[a_move] == BoardTile::None {board[a_move] = BoardTile::A; stalemate = false;}
        if let Some(winner) = has_won(&board) {return winner;}
        let b_move = b.get_move(&board, false);
        if board[b_move] == BoardTile::None {board[b_move] = BoardTile::B; stalemate = false;}
        if let Some(winner) = has_won(&board) {return winner;}
        if stalemate {return MatchWinner::Tie;}
    }
}

fn print_board(board: &[BoardTile; 27]) {
    println!();
    for l in 0..3 {
        for r in 0..3 {
            for c in 0..3 {
                print!("{} ", match board[l*9+r*3+c] {
                    BoardTile::None => "_",
                    BoardTile::A => "x",
                    BoardTile::B => "o"
                });
            }
            print!("   ");
            for c in 0..3 {
                print!("{} ", l*9+r*3+c);
            }
            println!();
        }
        println!();
    }
}

fn noughts_and_crosses_genetic() {
    let mut population: Vec<NACSolver> = (0..POPULATION_SIZE).map(|_| NACSolver::random()).collect();
    let mut rng = thread_rng();
    let mut loop_n = 0;
    let mut ties_mode = true;
    let solution_i = loop {
        let mut game_order = (0..POPULATION_SIZE).map(|n| (n, rng.gen::<u8>())).collect::<Vec<_>>();
        game_order.sort_unstable_by_key(|(_, k)| *k);
        let mut kill_marks = Vec::with_capacity(250);
        let mut survivors = Vec::with_capacity(500);
        let mut ties = 0;
        while let (Some((ai, _)), Some((bi, _))) = (game_order.pop(), game_order.pop()) {
            let a = &population[ai];
            let b = &population[bi];
            let winner = fight(a, b);
            match winner {
                MatchWinner::A => {kill_marks.push(bi); for _ in 0..5 {survivors.push(ai)};},
                MatchWinner::B => {kill_marks.push(ai); survivors.push(bi);},
                _ => {
                    if rng.gen_bool(0.5) {
                        kill_marks.push(ai); survivors.push(bi);
                    } else {
                        kill_marks.push(bi); survivors.push(ai);
                    }
                    if ties_mode {
                        ties += 1;
                    } else {ties -= 1;}
                }
            };
            if !ties_mode {
                ties += 1;
            }
        }
        for i in kill_marks {
            let parent_a = &population[survivors[rng.gen_range(0..survivors.len())]];
            let parent_b= &population[survivors[rng.gen_range(0..survivors.len())]];
            population[i] = NACSolver::cross(parent_a, parent_b);
        }
        loop_n += 1;
        if loop_n % 1000 == 0 {println!("{ties}");}
        // if ties_mode && ties == 0 {
            // ties_mode = false;
            // println!("okay, 0 random ties");
        // };
        // if !ties_mode && ties == POPULATION_SIZE/2 {
            // println!("done!");
            // break survivors[0];
        // }
        if loop_n == 10_000 {break survivors[0];}
    };
    let solution = &population[solution_i];
    'games: loop {
        println!("---------------");
        let mut board = [BoardTile::None; 27];
        loop {
            let ai_move = solution.get_move(&board, true);
            if board[ai_move] == BoardTile::None {
                board[ai_move] = BoardTile::A;
                println!("AI went on {ai_move}");
            } else {println!("AI forfeited turn.");}
            print_board(&board);
            if let Some(winner) = has_won(&board) {
                winner.announce(); continue 'games;
            };
            print!("Move: ");
            let mut inp = String::new();
            stdin().read_line(&mut inp).unwrap();
            let b_move = inp.trim().parse::<usize>().unwrap();
            board[b_move] = BoardTile::B;
            print_board(&board);
            if let Some(winner) = has_won(&board) {
                winner.announce(); continue 'games;
            };
        }
    }
}

fn main() {
    noughts_and_crosses_genetic();
}