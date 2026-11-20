"""Data providers and local cache access, imported only when requested."""
from importlib import import_module

_EXPORTS = {
    'IKlineProvider': ('data.base.interfaces', 'IKlineProvider'),
    'IQuoteProvider': ('data.base.interfaces', 'IQuoteProvider'),
    'ISymbolProvider': ('data.base.interfaces', 'ISymbolProvider'),
    'IIndustryProvider': ('data.base.interfaces', 'IIndustryProvider'),
    # DataLoader/CachedDataSource 随旧研究管道归档（2026-11-04 清理，
    # 快照在 archive/2026-10-30-before-reset/source/data/base/data_loader.py）
    'WindDataSource': ('data.wind', 'WindDataSource'),
    'AKShareDataSource': ('data.akshare', 'AKShareDataSource'),
    'WindDBDataSource': ('data.wind_db', 'WindDBDataSource'),
    'IndustryDataSource': ('data.wind_db', 'WindDBDataSource'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, member = _EXPORTS[name]
    value = getattr(import_module(module), member)
    globals()[name] = value
    return value
