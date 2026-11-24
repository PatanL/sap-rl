//! Super Auto Pets (Turtle Pack) battle engine.
//!
//! A seed-exact Rust port of the Python reference engine (engine-v31). The
//! acceptance test is `cargo run --release --bin golden`, which replays every
//! row of `golden/v31-340.jsonl` and requires the per-round trace, the
//! outcome and the next random draw to match the reference exactly.

pub mod batch;
pub mod canon;
pub mod data;
pub mod engine;
pub mod env;
pub mod envdata;
pub mod ffi;
pub mod obs;
pub mod rng;
pub mod tape;

pub use data::{Data, Status};
pub use engine::{simulate, Boards, EngineError, Fight, Outcome, PetSpec, SlotView};
