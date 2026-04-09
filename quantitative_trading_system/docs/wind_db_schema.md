# Wind 量化数据库数据结构

> 参考文档：Wind资讯量化研究数据库.pdf
>
> **重要说明**：以下表名和字段名均来自PDF文档中的正式定义。
> PDF中的表名格式为 "中文名称-AShareXXX"，表示这是Oracle数据库中的实际表名。

## 1. 股票日行情 (AShareEODPrices)

> 参考: Wind资讯量化研究数据库.pdf - 5.23 中国A 股日行情-AShareEODPrices
> 更新频率: 16:00

| 字段名 | 数据类型 | 中文名 | 备注 |
|--------|----------|--------|------|
| OBJECT_ID | VARCHAR2(100) | 对象ID | |
| s_info_windcode | VARCHAR2(40) | Wind代码 | 如 300308.SZ |
| trade_dt | VARCHAR2(8) | 交易日期 | YYYYMMDD |
| crncy_code | VARCHAR2(10) | 货币代码 | CNY 人民币 |
| s_dq_preclose | NUMBER(20,4) | 昨收盘价(元) | |
| s_dq_open | NUMBER(20,4) | 开盘价(元) | |
| s_dq_high | NUMBER(20,4) | 最高价(元) | |
| s_dq_low | NUMBER(20,4) | 最低价(元) | |
| s_dq_close | NUMBER(20,4) | 收盘价(元) | |
| s_dq_change | NUMBER(20,4) | 涨跌(元) | |
| s_dq_pctchange | NUMBER(20,4) | 涨跌幅(%) | |
| s_dq_volume | NUMBER(20,4) | 成交量(手) | |
| s_dq_amount | NUMBER(20,4) | 成交金额(千元) | |
| s_dq_adjpreclose | NUMBER(20,4) | 复权昨收盘价(元) | 昨收盘价*复权因子 |
| s_dq_adjopen | NUMBER(20,4) | 复权开盘价(元) | 开盘价*复权因子 |
| s_dq_adjhigh | NUMBER(20,4) | 复权最高价(元) | 最高价*复权因子 |
| s_dq_adjlow | NUMBER(20,4) | 复权最低价(元) | 最低价*复权因子 |
| s_dq_adjclose | NUMBER(20,4) | 复权收盘价(元) | 收盘价*复权因子 |
| s_dq_adjfactor | NUMBER(20,6) | 复权因子 | 初始值为1 |
| S_dq_avgprice | NUMBER(20,4) | 均价(VWAP) | 成交金额/成交量 |
| s_dq_tradestatus | VARCHAR2(10) | 交易状态 | |

## 2. 行情衍生指标 (AShareEODDerivativeIndicator)

> 参考: Wind资讯量化研究数据库.pdf - 5.28 中国A 股行情衍生指标-AShareEODDerivativeIndicator
> 更新频率: 16:00, 17:00

| 字段名 | 数据类型 | 中文名 | 备注 |
|--------|----------|--------|------|
| OBJECT_ID | VARCHAR2(100) | 对象ID | |
| s_info_windcode | VARCHAR2(40) | Wind代码 | |
| trade_dt | VARCHAR2(8) | 交易日期 | |
| crncy_code | VARCHAR2(10) | 货币代码 | CNY 人民币 |
| s_val_mv | NUMBER(20,4) | 总市值 | 收盘价*总股本 |
| s_dq_mv | NUMBER(20,4) | 流通市值 | |
| s_pq_high_52w | NUMBER(20,4) | 52周最高价 | |
| s_pq_low_52w | NUMBER(20,4) | 52周最低价 | |
| s_val_pe | NUMBER(20,4) | 市盈率(PE) | 市值/净利润 |
| s_val_pb_new | NUMBER(20,4) | 市净率(PB) | 市值/净资产(LF) |
| s_val_pe_ttm | NUMBER(20,4) | 市盈率(PE,TTM) | 市值/净利润TTM |
| s_val_pcf_ocf | NUMBER(20,4) | 市现率(PCF,经营现金流) | |
| s_val_pcf_ocfttm | NUMBER(20,4) | 市现率(PCF,经营现金流TTM) | |
| s_val_pcf_ncf | NUMBER(20,4) | 市现率(PCF,现金净流量) | |

## 3. 资产负债表 (AShareBalanceSheet)

> 参考: Wind资讯量化研究数据库.pdf - 5.43 中国A 股资产负债-AShareBalanceSheet
> 更新频率: 07:00, 09:00, 22:00
> 单位: 元

### 主要字段

| 字段名 | 数据类型 | 中文名 | 备注 |
|--------|----------|--------|------|
| OBJECT_ID | VARCHAR2(100) | 对象ID | |
| s_info_windcode | VARCHAR2(40) | Wind代码 | |
| WIND_CODE | VARCHAR2(40) | Wind代码 | |
| ann_dt | VARCHAR2(8) | 公告日期 | |
| report_period | VARCHAR2(8) | 报告期 | |
| statement_type | VARCHAR2(10) | 报表类型 | 408001000:合并报表; 408006000:母公司报表 |
| crncy_code | VARCHAR2(10) | 货币代码 | CNY |
| monetary_cap | NUMBER(20,4) | 货币资金 | |
| tot_cur_assets | NUMBER(20,4) | 流动资产合计 | |
| tot_non_cur_assets | NUMBER(20,4) | 非流动资产合计 | |
| tot_assets | NUMBER(20,4) | 资产总计 | |
| tot_cur_liab | NUMBER(20,4) | 流动负债合计 | |
| tot_non_cur_liab | NUMBER(20,4) | 非流动负债合计 | |
| tot_liabilities | NUMBER(20,4) | 负债合计 | |
| minority_int | NUMBER(20,4) | 少数股东权益 | |
| tot_shrhldr_equity | NUMBER(20,4) | 股东权益合计(不含少数股东) | |
| tot_shrhldr_equity_incl_min_int | NUMBER(20,4) | 股东权益合计(含少数股东) | |
| debt_asset_ratio | NUMBER(20,4) | 资产负债率 | % |

## 4. 利润表 (AShareIncome)

> 参考: Wind资讯量化研究数据库.pdf - 5.44 中国A 股利润表-AShareIncome
> 更新频率: 07:00, 09:00, 22:00
> 单位: 元

### 主要字段

| 字段名 | 数据类型 | 中文名 | 备注 |
|--------|----------|--------|------|
| OBJECT_ID | VARCHAR2(100) | 对象ID | |
| s_info_windcode | VARCHAR2(40) | Wind代码 | |
| WIND_CODE | VARCHAR2(40) | Wind代码 | |
| ann_dt | VARCHAR2(8) | 公告日期 | |
| report_period | VARCHAR2(8) | 报告期 | |
| statement_type | VARCHAR2(10) | 报表类型 | 408001000:合并报表; 408006000:母公司报表 |
| crncy_code | VARCHAR2(10) | 货币代码 | CNY |
| tot_oper_rev | NUMBER(20,4) | 营业总收入 | |
| oper_rev | NUMBER(20,4) | 营业收入 | |
| tot_profit | NUMBER(20,4) | 利润总额 | |
| net_profit | NUMBER(20,4) | 净利润 | |
| net_profit_under_minority_int | NUMBER(20,4) | 净利润(含少数股东损益) | |
| net_profit_excl_minority_int | NUMBER(20,4) | 净利润(归母净利润) | |
| s_fa_eps_basic | NUMBER(20,4) | 基本每股收益 | |
| s_fa_eps_diluted | NUMBER(20,4) | 稀释每股收益 | |

## 5. 现金流量表 (AShareCashFlow)

> 参考: Wind资讯量化研究数据库.pdf - 5.45 中国A 股现金流量表-AShareCashFlow
> 更新频率: 07:00, 09:00, 22:00
> 单位: 元

### 主要字段

| 字段名 | 数据类型 | 中文名 | 备注 |
|--------|----------|--------|------|
| OBJECT_ID | VARCHAR2(100) | 对象ID | |
| s_info_windcode | VARCHAR2(40) | Wind代码 | |
| WIND_CODE | VARCHAR2(40) | Wind代码 | |
| ann_dt | VARCHAR2(8) | 公告日期 | |
| report_period | VARCHAR2(8) | 报告期 | |
| statement_type | VARCHAR2(10) | 报表类型 | 408001000:合并报表; 408006000:母公司报表 |
| crncy_code | VARCHAR2(10) | 货币代码 | CNY |
| monetary_cap | NUMBER(20,4) | 期末现金及等价物 | |
| net_act_cash_frm_operat_act | NUMBER(20,4) | 经营活动现金流量净额 | |
| net_act_cash_frm_invest_act | NUMBER(20,4) | 投资活动现金流量净额 | |
| net_act_cash_frm_finan_act | NUMBER(20,4) | 筹资活动现金流量净额 | |
| s_fa_fcff | NUMBER(20,4) | 企业自由现金流量(FCFF) | |

## 6. 财务指标 (AShareFinancialIndicator)

> 参考: Wind资讯量化研究数据库.pdf - 5.46 中国A 股财务指标-AShareFinancialIndicator
> 更新频率: 09:00, 22:00

### 主要字段

| 字段名 | 数据类型 | 中文名 | 备注 |
|--------|----------|--------|------|
| OBJECT_ID | VARCHAR2(100) | 对象ID | |
| s_info_windcode | VARCHAR2(40) | Wind代码 | |
| WIND_CODE | VARCHAR2(40) | Wind代码 | |
| ann_dt | VARCHAR2(8) | 公告日期 | |
| report_period | VARCHAR2(8) | 报告期 | |
| crncy_code | VARCHAR2(10) | 货币代码 | 默认为人民币 |
| s_fa_grossmargin | NUMBER(20,4) | 毛利率 | % |
| s_fa_netprofitmargin | NUMBER(20,4) | 净利率 | % |
| s_fa_roe | NUMBER(20,4) | 净资产收益率(ROE) | % |
| s_fa_roe_diluted | NUMBER(20,4) | 稀释净资产收益率 | % |
| s_fa_roe_avg5y | NUMBER(20,4) | 平均5年净资产收益率 | % |
| s_fa_debtassetratio | NUMBER(20,4) | 资产负债率 | % |
| s_fa_currentratio | NUMBER(20,4) | 流动比率 | |
| s_fa_quickratio | NUMBER(20,4) | 速动比率 | |
| s_fa_cashratio | NUMBER(20,4) | 现金比率 | |
| s_fa_eps_basic | NUMBER(20,4) | 基本每股收益 | |
| s_fa_eps_diluted | NUMBER(20,4) | 稀释每股收益 | |
| s_fa_bps | NUMBER(20,4) | 每股净资产 | |
| s_fa_ocfps | NUMBER(20,4) | 每股经营活动现金流量净额 | |
| s_fa_fcff | NUMBER(20,4) | 企业自由现金流量(FCFF) | |
| s_fa_fcfe | NUMBER(20,4) | 股权自由现金流量(FCFE) | |

## 7. 分红数据 (AShareDividend)

> 参考: Wind资讯量化研究数据库.pdf - 5.17 中国A 股分红-AShareDividend
> 更新频率: 07:00, 08:30, 22:00

| 字段名 | 数据类型 | 中文名 | 备注 |
|--------|----------|--------|------|
| OBJECT_ID | VARCHAR2(100) | 对象ID | |
| s_info_windcode | VARCHAR2(40) | Wind代码 | |
| WIND_CODE | VARCHAR2(40) | Wind代码 | |
| progress | VARCHAR2(10) | 方案进度 | 1:董事会预案; 2:股东大会通过; 3:实施 |
| stk_dvd_per_sh | NUMBER(20,4) | 每股送转股 | |
| cash_dvd_per_sh_pre_tax | NUMBER(20,4) | 每股现金分红(含税) | |
| cash_dvd_per_sh_after_tax | NUMBER(20,4) | 每股现金分红(扣税) | |
| eqy_record_dt | VARCHAR2(8) | 股权登记日 | |
| ex_dt | VARCHAR2(8) | 除权除息日 | |
| pay_dt | VARCHAR2(8) | 派息日 | |

## 8. 股东数据 (AShareHolderNumber)

> 参考: Wind资讯量化研究数据库.pdf - 5.35 中国A 股股东户数-AShareHolderNumber
> 更新频率: 09:00, 22:00

| 字段名 | 数据类型 | 中文名 | 备注 |
|--------|----------|--------|------|
| OBJECT_ID | VARCHAR2(100) | 对象ID | |
| s_info_windcode | VARCHAR2(40) | Wind代码 | |
| holder_date | VARCHAR2(8) | 截止日期 | |
| holder_num | NUMBER(20,4) | 股东户数 | |
| holder_num_f | NUMBER(20,4) | 流通股东户数 | |
| holder_avg_market_cap | NUMBER(20,4) | 户均市值 | |
| inst_holder_ratio | NUMBER(20,4) | 机构持仓比例 | % |

## 9. 股票基本信息 (AShareDescription)

> 参考: Wind资讯量化研究数据库.pdf - 5.1 中国A 股基础信息-AShareDescription

| 字段名 | 数据类型 | 中文名 | 备注 |
|--------|----------|--------|------|
| s_info_windcode | VARCHAR2(40) | Wind代码 | |
| s_info_code | VARCHAR2(40) | 证券代码 | |
| s_info_name | VARCHAR2(50) | 证券名称 | |
| s_info_compname | VARCHAR2(100) | 公司中文名称 | |
| s_info_compnameeng | VARCHAR2(100) | 公司英文名称 | |
| s_info_isincode | VARCHAR2(40) | ISIN代码 | |

## 10. 行业分类 (AShareIndustriesClass)

> 参考: Wind资讯量化研究数据库.pdf - 5.4 中国A 股行业分类-AShareIndustriesClass

| 字段名 | 数据类型 | 中文名 | 备注 |
|--------|----------|--------|------|
| s_info_windcode | VARCHAR2(40) | Wind代码 | |
| wind_ind_code | VARCHAR2(50) | Wind行业代码 | |
| wind_ind_name | VARCHAR2(100) | Wind行业名称 | |

## 数据库表名速查

| 数据类型 | 表名 | PDF章节 |
|---------|------|---------|
| 日行情 | AShareEODPrices | 5.23 |
| 行情衍生指标 | AShareEODDerivativeIndicator | 5.28 |
| 资产负债表 | AShareBalanceSheet | 5.43 |
| 利润表 | AShareIncome | 5.44 |
| 现金流量表 | AShareCashFlow | 5.45 |
| 财务指标 | AShareFinancialIndicator | 5.46 |
| 分红 | AShareDividend | 5.17 |
| 股东 | AShareHolderNumber | 5.35 |
| 股票信息 | AShareDescription | 5.1 |
| 行业分类 | AShareIndustriesClass | 5.4 |

## 使用说明

1. **股票代码字段**: Wind代码格式为 `600000.SH`（上海）、`000001.SZ`（深圳）
2. **日期格式**: 所有日期字段格式为 `YYYYMMDD`（字符串）
3. **报表类型**: `statement_type = '408001000'` 表示合并报表
4. **货币代码**: `crncy_code = 'CNY'` 表示人民币
5. **数据单位**: 财务报表金额单位通常为元（除特别注明外）
