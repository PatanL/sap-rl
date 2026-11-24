#!/usr/bin/env node
"use strict";
// Run SAP-Calculator on battle configs and return, per config, the outcome
// counts and every distinct final board (surviving attack/health per side,
// front to back). One fresh process per config: reusing the bundle across
// boards can change results.
//
// Usage: calc_states.js <SAP-Calculator checkout> [--log]   (configs on stdin)
const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");

const calculatorRoot = process.argv[2];
const wantLog = process.argv.includes("--log");
const requests = JSON.parse(fs.readFileSync(0, "utf8"));

function clean(message) {
  return String(message || "").replace(/<[^>]+>/g, "");
}

function finalBoard(logs) {
  // Last board line that lists slots, e.g. "P5 (3/1/2xp) ___ (-/-) | O1 (5/7/0xp) ...".
  for (let i = logs.length - 1; i >= 0; i--) {
    const text = clean(logs[i].message);
    if (logs[i].type !== "board" || !text.includes("|")) continue;
    const [left, right] = text.split("|");
    const parse = side => [...side.matchAll(/[PO]\d \((\d+)\/(\d+)\//g)].map(m => [Number(m[1]), Number(m[2])]);
    // Player slots print back to front; opponent slots front to back.
    return { player: parse(left).reverse(), opponent: parse(right) };
  }
  return null;
}

function simulate(config) {
  const origLog = console.log;
  console.log = () => {};
  const simulation = require(path.join(calculatorRoot, "simulation/dist/index.js"));
  const result = simulation.runHeadlessSimulation(
    { ...config, logsEnabled: true, maxLoggedBattles: config.simulationCount },
    { includeBattles: true, enableLogs: true });
  console.log = origLog;
  const states = new Map();
  for (const battle of result.battles || []) {
    const board = finalBoard(battle.logs || []);
    const key = JSON.stringify([battle.winner, board]);
    states.set(key, (states.get(key) || 0) + 1);
  }
  const out = {
    playerWins: result.playerWins, opponentWins: result.opponentWins, draws: result.draws,
    states: [...states.entries()].map(([key, count]) => ({ ...JSON.parse(key).reduce((a, v, i) => (i ? { ...a, board: v } : { winner: v }), {}), count })),
  };
  if (wantLog && result.battles && result.battles[0]) {
    out.log = result.battles[0].logs.map(l => `${l.type} | ${clean(l.message)}`);
  }
  return out;
}

if (requests.length === 1 && process.env.CALC_CHILD) {
  process.stdout.write(JSON.stringify([simulate(requests[0])]));
} else {
  const results = requests.map(config => {
    const args = [__filename, calculatorRoot].concat(wantLog ? ["--log"] : []);
    const output = execFileSync(process.execPath, args, {
      input: JSON.stringify([config]), encoding: "utf8", maxBuffer: 1 << 28,
      env: { ...process.env, CALC_CHILD: "1" },
    });
    return JSON.parse(output)[0];
  });
  process.stdout.write(JSON.stringify(results));
}
