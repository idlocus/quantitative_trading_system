"""Execution, accounting and reconciliation without an active strategy."""
from .broker import Order, Broker, PaperBroker, Fill
from .ledger import Ledger
from .risk_gate import RiskGate, GateConfig, RiskDecision
from .reconcile import Reconciler, BrokerStatement, ReconReport

__all__ = ['Order', 'Fill', 'Broker', 'PaperBroker', 'Ledger', 'RiskGate',
           'GateConfig', 'RiskDecision', 'Reconciler', 'BrokerStatement',
           'ReconReport', 'QMTBroker', 'make_broker', 'load_lot_map']


def __getattr__(name):
    if name == 'QMTBroker':
        from .qmt_broker import QMTBroker
        return QMTBroker
    raise AttributeError(name)


def make_broker(mode, lot_map=None, **kwargs):
    """Expose simulation only while the workspace has no accepted strategy."""
    if mode == 'paper':
        return PaperBroker(lot_map=lot_map)
    if mode == 'qmt':
        raise RuntimeError('QMT execution is disabled in the strategy-free workspace')
    raise ValueError(f'Unknown broker mode: {mode}')


def load_lot_map(ds):
    """Read board lot metadata without selecting a trading strategy."""
    from data.board_params import BoardParamsProvider
    return BoardParamsProvider(ds).lot_map
