// Honjo Masamune — Core IR ("Cut-IR", §8)
// A typed, back-end-agnostic sequence of cut instructions over a contact-graph
// state. Both the TypeScript (render-pass) and Rust (max-flow) back ends
// consume exactly this IR; only the realisation of a cut differs.

import { TyName } from "./types.js";
import { RelOp } from "./ast.js";

export type IRValueRef = { kind: "reg"; id: number } | { kind: "lit"; value: number; floor: number } | { kind: "str"; value: string };

export type IRCond = { left: IROperand; op: RelOp; right: IROperand };
export type IROperand =
  | { kind: "reg"; id: number }
  | { kind: "lit"; value: number; floor: number }
  | { kind: "field"; reg: number; field: string }   // e.g. W.valence, delta
  | { kind: "name"; name: string };                  // bare measured field (delta, vacancy)

export type Admit =
  | { kind: "converge" }
  | { kind: "diverge" }
  | { kind: "cond"; cond: IRCond };

// Every instruction writes its result to a register `dst` (except Floor/Assert/Obs).
export type Instr =
  | { op: "Floor"; value: number }
  | { op: "Ind"; dst: number; z: IROperand; ty: TyName; floor: number }            // individuate (cut Z)
  | { op: "Bnd"; dst: number; a: number; b: number; guard?: IRCond; ty: TyName; floor: number } // bond (a ~ b)
  | { op: "Cls"; dst: number; central: number; args: number[]; ty: TyName; floor: number }      // close
  | { op: "Prp"; dst: number; item: number; process: number; reps?: string[]; admit: Admit; ty: TyName; floor: number } // propagate / track
  | { op: "Obs"; reg: number; as?: number }          // force measurement
  | { op: "Assert"; cond: IRCond; message?: string }
  | { op: "Call"; dst: number; name: string[]; args: IROperand[]; ty: TyName; floor: number }
  | { op: "Mov"; dst: number; src: IROperand };       // bind passthrough for non-cut exprs

export interface IRProgram {
  instrs: Instr[];
  // maps source binding names to the register holding their value
  nameToReg: Map<string, number>;
  floor: number; // ambient floor at end (for reporting)
}
