"""
DERIV ACCUMULATOR BOT — 1HZ10V
================================
Symbol   : 1HZ10V (configurable)
Contract : ACCU — 1-tick accumulator
           Stake grows by growth_rate% if price stays in range
           for exactly ONE tick. Auto-settles immediately.

Strategy
--------
  At 1 tick, the contract is essentially:
    · Win  → price stayed in range this tick → collect growth_rate% of stake
    · Lose → price breached the range        → lose stake

  Entry only when σ is VERY CALM (ratio < ENTRY_CALM threshold).
  Growth rate selected by calm depth:
    · extremely calm (ratio < 0.50 × avg) → 5% growth
    · very calm      (ratio < 0.60 × avg) → 4% growth
    · calm           (ratio < 0.70 × avg) → 3% growth
    · above threshold                     → skip

  No take-profit sell needed — 1-tick contracts auto-settle.
  High frequency — re-enters immediately after each settlement.

  Monitor data basis (1HZ10V, ~6300 ticks):
    · σ avg = 0.100, range 0.06–0.14
    · 99.9% time in NORMAL band
    · At 1-tick, knockout risk is minimal during calm windows

Risk
----
  Flat stake $2.00 per trade. No martingale.
  Circuit breaker: 5 consecutive knockouts → 5 min pause.
  Session target: +$10 | Session stop: -$20
"""

import asyncio
import json
import math
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Optional

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed, ConnectionClosedError, ConnectionClosedOK,
    )
except ImportError:
    sys.exit("websockets not installed — run: pip install websockets")


# ============================================================================
# CONFIGURATION
# ============================================================================

def _env(key, default):
    val = os.environ.get(key)
    if val is None:
        return default
    if isinstance(default, bool):
        return val.lower() in ("1", "true", "yes")
    if isinstance(default, float):
        return float(val)
    if isinstance(default, int):
        return int(val)
    return val


CONFIG = {
    "api_token":            _env("DERIV_API_TOKEN", "REPLACE_WITH_YOUR_TOKEN"),
    "app_id":               _env("DERIV_APP_ID", 1089),
    "symbol":               _env("SYMBOL", "1HZ10V"),
    "currency":             "USD",

    # Volatility window for σ calculation
    "vol_window":           _env("VOL_WINDOW", 50),
    "min_warmup":           _env("MIN_WARMUP", 60),

    # Entry calm thresholds (σ / session_avg ratio)
    "entry_extremely_calm": _env("ENTRY_EXTREMELY_CALM", 0.50),  # → 5% growth
    "entry_very_calm":      _env("ENTRY_VERY_CALM",      0.60),  # → 4% growth
    "entry_calm":           _env("ENTRY_CALM",           0.70),  # → 3% growth

    # Cooldown ticks between trades
    "cooldown_ticks":       _env("COOLDOWN_TICKS", 2),

    # Risk — flat stake, no martingale
    "stake":                _env("STAKE", 2.00),
    "target_profit":        _env("TARGET_PROFIT", 10.0),
    "stop_loss":            _env("STOP_LOSS", 20.0),

    # Circuit breaker
    "cb_limit":             _env("CB_LIMIT", 5),
    "cb_pause_secs":        _env("CB_PAUSE", 300),

    # Resilience
    "lock_timeout":         _env("LOCK_TIMEOUT", 30),
    "buy_retries":          _env("BUY_RETRIES", 8),
    "reconnect_min":        _env("RECONNECT_MIN", 2),
    "reconnect_max":        _env("RECONNECT_MAX", 60),
    "ws_ping":              _env("WS_PING", 30),
    "orphan_attempts":      _env("ORPHAN_ATTEMPTS", 4),
    "orphan_interval":      _env("ORPHAN_INTERVAL", 3),
}


# ============================================================================
# HELPERS
# ============================================================================

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _log(tag, msg):
    print(f"[{_ts()}] [{tag}] {msg}", flush=True)

def _jlog(obj):
    print(json.dumps(obj), flush=True)


# ============================================================================
# VOLATILITY ENGINE
# ============================================================================

class VolEngine:
    def __init__(self, cfg):
        self.cfg           = cfg
        self.prices        = deque(maxlen=cfg["vol_window"] + 2)
        self.moves         = deque(maxlen=cfg["vol_window"])
        self.tick_n        = 0
        self.sigma_history = deque(maxlen=500)

    def add_tick(self, price: float):
        if self.prices:
            self.moves.append(abs(price - self.prices[-1]))
        self.prices.append(price)
        self.tick_n += 1

    def is_ready(self) -> bool:
        return self.tick_n >= self.cfg["min_warmup"]

    def sigma(self) -> float:
        if len(self.moves) < 5:
            return 0.0
        moves = list(self.moves)
        mu    = sum(moves) / len(moves)
        var   = sum((x - mu) ** 2 for x in moves) / len(moves)
        s     = math.sqrt(var)
        self.sigma_history.append(s)
        return s

    def session_avg_sigma(self) -> float:
        if not self.sigma_history:
            return 0.100   # fallback from 1HZ10V monitor data
        return sum(self.sigma_history) / len(self.sigma_history)

    def evaluate(self):
        """
        Returns (should_trade, growth_rate, sigma, ratio).
        growth_rate: 0.03 / 0.04 / 0.05
        """
        if not self.is_ready():
            return False, 0, 0.0, 0.0

        s   = self.sigma()
        avg = self.session_avg_sigma()

        if avg == 0:
            return False, 0, s, 0.0

        ratio = s / avg

        if ratio < self.cfg["entry_extremely_calm"]:
            return True, 0.05, s, ratio
        if ratio < self.cfg["entry_very_calm"]:
            return True, 0.04, s, ratio
        if ratio < self.cfg["entry_calm"]:
            return True, 0.03, s, ratio

        return False, 0, s, ratio


# ============================================================================
# RISK MANAGER — flat stake
# ============================================================================

class RiskManager:
    def __init__(self, cfg):
        self.stake         = cfg["stake"]
        self.target_profit = cfg["target_profit"]
        self.stop_loss     = cfg["stop_loss"]
        self.total_profit  = 0.0
        self.wins          = 0
        self.losses        = 0
        self.loss_streak   = 0

    def record_win(self, profit: float):
        self.wins         += 1
        self.total_profit += profit
        self.loss_streak   = 0
        _log("WIN", f"+${profit:.4f} | total P&L ${self.total_profit:+.4f}")
        self._stats()

    def record_loss(self, loss: float):
        self.losses       += 1
        self.total_profit += loss
        self.loss_streak  += 1
        _log("KNOCKOUT",
             f"-${abs(loss):.2f} | streak={self.loss_streak} | "
             f"total P&L ${self.total_profit:+.4f}")
        self._stats()

    def can_trade(self) -> bool:
        if self.total_profit >= self.target_profit:
            _log("RISK",
                 f"Target profit reached (${self.total_profit:.4f}) — stopping")
            return False
        if self.total_profit <= -self.stop_loss:
            _log("RISK",
                 f"Stop-loss hit (${self.total_profit:.4f}) — stopping")
            return False
        return True

    def _stats(self):
        total = self.wins + self.losses
        wr    = (self.wins / total * 100) if total else 0.0
        print(f"\n{'='*55}", flush=True)
        print(f"  {total} trades | W:{self.wins} L:{self.losses} | WR:{wr:.1f}%",
              flush=True)
        print(f"  P&L ${self.total_profit:+.4f} | stake ${self.stake:.2f}",
              flush=True)
        print(f"{'='*55}\n", flush=True)
        _jlog({
            "type":   "stats",
            "trades": total,
            "wins":   self.wins,
            "losses": self.losses,
            "wr":     round(wr, 1),
            "pnl":    round(self.total_profit, 4),
            "stake":  self.stake,
            "ts":     _ts(),
        })


# ============================================================================
# DERIV CLIENT
# ============================================================================

class DerivClient:
    def __init__(self, cfg):
        self.cfg         = cfg
        self.endpoint    = (
            f"wss://ws.derivws.com/websockets/v3?app_id={cfg['app_id']}"
        )
        self.ws          = None
        self._send_queue = None
        self._inbox      = None
        self._send_task  = None
        self._recv_task  = None

    async def connect(self) -> bool:
        _log("WS", f"Connecting → {self.endpoint}")
        self.ws = await websockets.connect(
            self.endpoint,
            ping_interval=self.cfg["ws_ping"],
            ping_timeout=20,
            close_timeout=10,
        )
        self._send_queue = asyncio.Queue()
        self._inbox      = asyncio.Queue()
        self._start_io()
        await self._send({"authorize": self.cfg["api_token"]})
        resp = await self._recv_type("authorize", timeout=15)
        if not resp or "error" in resp:
            err = (resp or {}).get("error", {}).get("message", "timeout")
            _log("AUTH", f"Failed: {err}")
            return False
        auth = resp.get("authorize", {})
        _log("AUTH",
             f"OK | {auth.get('loginid','?')} | "
             f"Balance: ${auth.get('balance', 0):.2f}")
        return True

    def _start_io(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        self._send_task = asyncio.create_task(self._send_pump())
        self._recv_task = asyncio.create_task(self._recv_pump())

    async def _send_pump(self):
        while True:
            data, fut = await self._send_queue.get()
            try:
                await self.ws.send(json.dumps(data))
                if fut and not fut.done():
                    fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done():
                    fut.set_exception(exc)
            finally:
                self._send_queue.task_done()

    async def _recv_pump(self):
        try:
            async for raw in self.ws:
                try:
                    await self._inbox.put(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            _log("RECV", f"Error: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def close(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass

    async def _send(self, data):
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        await self._send_queue.put((data, fut))
        await fut

    async def receive(self, timeout=60):
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return {}

    async def _recv_type(self, msg_type, timeout=10):
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(
                    self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if "__disconnect__" in msg:
                await self._inbox.put(msg)
                return None
            if msg_type in msg or "error" in msg:
                return msg
            await self._inbox.put(msg)

    async def fetch_balance(self) -> Optional[float]:
        try:
            await self._send({"balance": 1})
            resp = await self._recv_type("balance", timeout=10)
            if resp and "balance" in resp:
                return float(resp["balance"]["balance"])
        except Exception as exc:
            _log("BALANCE", f"Fetch error: {exc}")
        return None

    async def subscribe_ticks(self) -> bool:
        sym = self.cfg["symbol"]
        await self._send({"ticks": sym, "subscribe": 1})
        resp = await self._recv_type("tick", timeout=10)
        if not resp or "error" in resp:
            err = (resp or {}).get("error", {}).get("message", "timeout")
            _log("TICK", f"Subscribe failed: {err}")
            return False
        _log("TICK", f"Subscribed to {sym}")
        return True

    async def place_accumulator(
            self, growth_rate: float, stake: float) -> Optional[str]:
        proposal_req = {
            "proposal":      1,
            "amount":        stake,
            "basis":         "stake",
            "contract_type": "ACCU",
            "currency":      self.cfg["currency"],
            "growth_rate":   growth_rate,
            "symbol":        self.cfg["symbol"],
        }
        await self._send(proposal_req)
        proposal = await self._recv_type("proposal", timeout=12)
        if not proposal or "error" in proposal:
            err = (proposal or {}).get("error", {}).get("message", "timeout")
            _log("PROPOSAL", f"Error: {err}")
            return None

        prop = proposal.get("proposal", {})
        pid  = prop.get("id")
        ask  = float(prop.get("ask_price", stake))

        if not pid:
            _log("PROPOSAL", "No proposal ID")
            return None

        win_amount = round(stake * growth_rate, 4)
        _log("PROPOSAL",
             f"ACCU  growth={growth_rate*100:.0f}%  ask=${ask:.2f}  "
             f"win_if_ok=+${win_amount:.4f}")

        buy_time    = time.time()
        contract_id = None
        await self._send({"buy": pid, "price": ask})

        for attempt in range(self.cfg["buy_retries"]):
            resp = await self._recv_type("buy", timeout=8)
            if resp is None:
                _log("BUY", f"No response (attempt {attempt + 1})")
                continue
            if "error" in resp:
                _log("BUY", f"Error: {resp['error'].get('message', '')}")
                return None
            contract_id = resp.get("buy", {}).get("contract_id")
            if contract_id:
                break

        if not contract_id:
            _log("BUY", "No contract_id — orphan recovery")
            contract_id = await self._recover_orphan(stake, buy_time)
            if contract_id:
                _log("BUY", f"Orphan recovered → {contract_id}")
            else:
                _log("BUY", "Orphan recovery failed")
                return None

        _log("TRADE",
             f"ACCU  ${stake:.2f}  growth={growth_rate*100:.0f}%  "
             f"contract={contract_id}")

        try:
            await self._send({
                "proposal_open_contract": 1,
                "contract_id":            contract_id,
                "subscribe":              1,
            })
        except Exception:
            pass

        return str(contract_id)

    async def _recover_orphan(self, stake, buy_time) -> Optional[str]:
        for attempt in range(self.cfg["orphan_attempts"]):
            await asyncio.sleep(self.cfg["orphan_interval"])
            try:
                await self._send({"profit_table": 1, "description": 1,
                                  "sort": "DESC", "limit": 5})
                resp = await self._recv_type("profit_table", timeout=10)
                if not resp or "error" in resp:
                    continue
                for tx in resp.get("profit_table", {}).get("transactions", []):
                    if (abs(float(tx.get("buy_price", 0)) - stake) < 0.01 and
                            float(tx.get("purchase_time", 0)) >= buy_time - 5):
                        return str(tx.get("contract_id"))
            except Exception as exc:
                _log("ORPHAN", f"Poll {attempt + 1} error: {exc}")
        return None

    async def poll_contract(self, contract_id) -> Optional[dict]:
        try:
            await self._send({"proposal_open_contract": 1,
                              "contract_id": contract_id})
            resp = await self._recv_type("proposal_open_contract", timeout=10)
            if resp and "proposal_open_contract" in resp:
                return resp["proposal_open_contract"]
        except Exception as exc:
            _log("POLL", f"Error: {exc}")
        return None


# ============================================================================
# MAIN BOT
# ============================================================================

class AccumulatorBot:
    def __init__(self):
        self.cfg    = CONFIG
        self.client = DerivClient(CONFIG)
        self.engine = VolEngine(CONFIG)
        self.risk   = RiskManager(CONFIG)

        self.tick_n:             int            = 0
        self._last_trade_tick:   int            = 0
        self.current_contract:   Optional[dict] = None
        self.waiting_for_result: bool           = False
        self.lock_since:         Optional[float] = None
        self._evaluating:        bool           = False
        self._balance_before:    Optional[float] = None
        self._cb_paused_until:   float          = 0.0
        self._stop:              bool           = False

    def _unlock(self, reason="manual"):
        if self.waiting_for_result:
            cid = (self.current_contract or {}).get("id", "?")
            _log("UNLOCK", f"Contract {cid} ({reason})")
        self.waiting_for_result = False
        self.current_contract   = None
        self.lock_since         = None
        self._evaluating        = False

    def _check_lock_timeout(self):
        if not self.waiting_for_result or self.lock_since is None:
            return
        if time.monotonic() - self.lock_since >= self.cfg["lock_timeout"]:
            _log("TIMEOUT", "Auto-unlocking after lock timeout")
            self._unlock("timeout")

    @staticmethod
    def _is_settled(data) -> bool:
        if data.get("is_sold") or data.get("is_expired"):
            return True
        for key in ("status", "contract_status"):
            if data.get(key, "").lower() in ("sold", "won", "lost"):
                return True
        return False

    async def handle_settlement(self, data) -> Optional[bool]:
        cid = str(data.get("contract_id", ""))
        if not self.current_contract or cid != self.current_contract["id"]:
            return None
        if not self._is_settled(data):
            return None

        bal_after  = await self.client.fetch_balance()
        api_profit = float(data.get("profit", 0))
        status     = data.get("status", "unknown")

        if bal_after is not None and self._balance_before is not None:
            actual = round(bal_after - self._balance_before, 4)
            _log("BALANCE",
                 f"Pre: ${self._balance_before:.2f} → "
                 f"Post: ${bal_after:.4f} | "
                 f"Actual: ${actual:+.4f} | API: ${api_profit:+.4f}")
        else:
            actual = api_profit

        print(f"\nRESULT  contract={cid}  status={status}  "
              f"profit=${actual:+.4f}", flush=True)

        if actual > 0:
            self.risk.record_win(actual)
        else:
            self.risk.record_loss(actual)
            if (self.risk.loss_streak > 0 and
                    self.risk.loss_streak % self.cfg["cb_limit"] == 0):
                pause = self.cfg["cb_pause_secs"]
                self._cb_paused_until = time.monotonic() + pause
                _log("BREAKER",
                     f"{self.cfg['cb_limit']} consecutive knockouts → "
                     f"pausing {pause}s ({pause // 60}m)")

        _jlog({
            "type":   "result",
            "cid":    cid,
            "status": status,
            "profit": actual,
            "pnl":    round(self.risk.total_profit, 4),
            "wins":   self.risk.wins,
            "losses": self.risk.losses,
            "ts":     _ts(),
        })

        self._balance_before = None
        self._unlock("settlement")
        return self.risk.can_trade()

    async def on_tick(self, price: float):
        self.tick_n += 1
        self.engine.add_tick(price)
        self._check_lock_timeout()

        if self.tick_n % 20 == 0:
            warmup_left = max(0, self.cfg["min_warmup"] - self.engine.tick_n)
            status = ("WAIT" if self.waiting_for_result else
                      f"WARMUP({warmup_left})" if warmup_left > 0 else "READY")
            print(f"\r  #{self.tick_n}  p={price:.5f}  {status}  {_ts()}",
                  end="", flush=True)

        if self.waiting_for_result or self._evaluating:
            return
        if not self.engine.is_ready():
            return
        if (self.tick_n - self._last_trade_tick) < self.cfg["cooldown_ticks"]:
            return

        self._evaluating = True
        try:
            await self._evaluate()
        finally:
            self._evaluating = False

    async def _evaluate(self):
        if self.waiting_for_result:
            return

        ok, growth_rate, sigma, ratio = self.engine.evaluate()
        avg = self.engine.session_avg_sigma()

        if not ok:
            return   # silent — too noisy to print every tick

        print(f"\n{'='*55}", flush=True)
        print(f"SIGNAL  #{self.tick_n}  {_ts()}", flush=True)
        print(f"  σ={sigma:.6f}  avg={avg:.6f}  ratio={ratio:.2f}", flush=True)
        print(f"  → ACCU 1-tick  growth={growth_rate*100:.0f}%  "
              f"stake=${self.cfg['stake']:.2f}  "
              f"win=+${self.cfg['stake']*growth_rate:.4f}", flush=True)
        print(f"{'='*55}", flush=True)

        now = time.monotonic()
        if now < self._cb_paused_until:
            remaining = self._cb_paused_until - now
            _log("BREAKER", f"Paused — {remaining:.0f}s remaining")
            return

        if not self.risk.can_trade():
            return

        stake = self.risk.stake
        bal   = await self.client.fetch_balance()
        if bal is not None:
            self._balance_before = bal
            _log("BALANCE", f"Pre-trade: ${bal:.2f}")
        else:
            self._balance_before = None

        contract_id = await self.client.place_accumulator(growth_rate, stake)

        if contract_id:
            self.current_contract = {
                "id":          contract_id,
                "stake":       stake,
                "growth_rate": growth_rate,
                "sigma":       sigma,
                "time":        datetime.now(),
            }
            self.waiting_for_result = True
            self.lock_since         = time.monotonic()
            self._last_trade_tick   = self.tick_n
            _log("LOCK", f"Waiting for 1-tick settlement on {contract_id}")
            _jlog({
                "type":        "trade",
                "cid":         contract_id,
                "growth_rate": growth_rate,
                "stake":       stake,
                "sigma":       round(sigma, 6),
                "ratio":       round(ratio, 3),
                "ts":          _ts(),
            })
        else:
            self._balance_before = None
            _log("TRADE", "Placement failed — ready for next signal")

    async def _reconnect(self) -> bool:
        delay   = self.cfg["reconnect_min"]
        attempt = 0
        while not self._stop:
            attempt += 1
            _log("RECONNECT", f"Attempt {attempt} in {delay}s...")
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.cfg["reconnect_max"])
            await self.client.close()
            self.client = DerivClient(self.cfg)
            try:
                if not await self.client.connect():
                    continue
                if not await self.client.subscribe_ticks():
                    continue
                if self.waiting_for_result and self.current_contract:
                    cid  = self.current_contract["id"]
                    data = await self.client.poll_contract(cid)
                    if data:
                        await self.handle_settlement(data)
                    if self.waiting_for_result:
                        await self.client._send({
                            "proposal_open_contract": 1,
                            "contract_id": cid,
                            "subscribe":   1,
                        })
                _log("RECONNECT", "OK")
                return True
            except Exception as exc:
                _log("RECONNECT", f"Error: {exc}")
        return False

    async def _console(self):
        loop = asyncio.get_event_loop()
        _log("CMD", "Commands: [s]tats  [u]nlock  [q]uit")
        while not self._stop:
            try:
                cmd = (await loop.run_in_executor(None, input)).strip().lower()
                if cmd == "s":
                    self.risk._stats()
                elif cmd == "u":
                    self._unlock("user command")
                elif cmd in ("q", "quit", "exit"):
                    self._stop = True
                    break
            except (EOFError, KeyboardInterrupt):
                break

    async def run(self):
        cfg = self.cfg
        print("\n" + "="*55, flush=True)
        print("  DERIV ACCUMULATOR BOT — 1-TICK", flush=True)
        print("="*55, flush=True)
        print(f"  Symbol      : {cfg['symbol']}", flush=True)
        print(f"  Contract    : ACCU 1-tick", flush=True)
        print(f"  Growth rate : 3–5% adaptive (calm σ only)", flush=True)
        print(f"  Stake       : ${cfg['stake']:.2f} flat", flush=True)
        print(f"  Entry gate  : σ < {cfg['entry_calm']}× session avg", flush=True)
        print(f"  Target      : +${cfg['target_profit']}  "
              f"Stop: -${cfg['stop_loss']}", flush=True)
        print(f"  Breaker     : {cfg['cb_limit']} knockouts → "
              f"{cfg['cb_pause_secs']}s pause", flush=True)
        print(f"  Warmup      : {cfg['min_warmup']} ticks", flush=True)
        print("="*55 + "\n", flush=True)

        if cfg["api_token"] in ("REPLACE_WITH_YOUR_TOKEN", ""):
            _log("ERROR", "Set DERIV_API_TOKEN before running")
            return

        if not await self.client.connect():
            return
        if not await self.client.subscribe_ticks():
            return

        _log("BOT", f"Live — warming up ({cfg['min_warmup']} ticks)...")
        console_task = asyncio.create_task(self._console())

        try:
            while not self._stop:
                response = await self.client.receive(timeout=60)

                if "__disconnect__" in response:
                    _log("WS", "Disconnected — reconnecting")
                    if not await self._reconnect():
                        break
                    continue

                if not response:
                    try:
                        await self.client.ws.ping()
                    except Exception:
                        _log("WS", "Ping failed — reconnecting")
                        if not await self._reconnect():
                            break
                    continue

                if "tick" in response:
                    quote = response["tick"].get("quote")
                    if quote is not None:
                        print()
                        await self.on_tick(float(quote))

                if "proposal_open_contract" in response:
                    result = await self.handle_settlement(
                        response["proposal_open_contract"])
                    if result is False:
                        break

                if "buy" in response:
                    result = await self.handle_settlement(response["buy"])
                    if result is False:
                        break

                if "transaction" in response:
                    tx = response["transaction"]
                    if "contract_id" in tx:
                        result = await self.handle_settlement({
                            "contract_id": tx.get("contract_id"),
                            "profit":      tx.get("profit", 0),
                            "status":      tx.get("action", ""),
                            "is_sold":     True,
                        })
                        if result is False:
                            break

        except KeyboardInterrupt:
            print("\n\nInterrupted", flush=True)
        except Exception as exc:
            print(f"\nUnhandled error: {exc}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            console_task.cancel()
            await self.client.close()
            print("\nFINAL STATS", flush=True)
            self.risk._stats()
            print("Goodbye", flush=True)


async def main():
    bot = AccumulatorBot()
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
