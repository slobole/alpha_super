# FinHacker S&P 500 Top-20 Market-Cap Extract

This folder stores research-only extracts from:

`https://www.finhacker.cz/en/top-20-sp-500-companies-by-market-cap/`

## Data Contract

- `finhacker_sp500_top20_market_cap_monthly.csv`
  - Long CSV intended for `strategy_mo_sp10_market_cap_rotation.py`.
  - Required columns: `date`, `symbol`, `market_cap`.
  - Extra audit columns: `company_name`, `market_cap_text`, `source_url`.

- `finhacker_sp500_top20_market_cap_annual_top20.csv`
  - Annual top-20 audit extract from the source page.
  - Not the preferred strategy input because annual rows are not a monthly PIT matrix.

## Caveat

This is not official S&P DJI constituent weight data and not Norgate PIT market-cap data.
It is a third-party compiled/backfilled market-cap source. Norgate must still be used
for PIT S&P 500 membership and historical prices.

The monthly CSV only covers companies linked from the annual top-20 page. That should
capture the dominant mega-cap names, but it is not a complete S&P 500 market-cap matrix.
If a company entered the true monthly top 10 but never appeared in any annual top-20 list,
this extract can miss it.
