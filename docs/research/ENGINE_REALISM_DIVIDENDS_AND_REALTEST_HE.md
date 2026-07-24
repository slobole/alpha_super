# ריאליזם במנוע: דיבידנדים, התאמות מחיר ו־RealTest

## TL;DR

המנוע הנוכחי **אינו שמרני באופן אחיד**. ברירת המחדל היא להשתמש ב־`CAPITALSPECIAL` עבור נכסים נסחרים וב־`TOTALRETURN` עבור בנצ'מרקים, אך מנוע `Strategy` המשותף אינו רושם דיבידנד במזומן ואינו מחייב ריבית מימון באופן כללי.

התוצאה תלויה בסוג האסטרטגיה:

- אסטרטגיית Long שמחזיקה דרך חלוקת דיבידנד נראית בדרך כלל חלשה מדי.
- אסטרטגיית Short נראית בדרך כלל טובה מדי, משום שאין חיוב דיבידנד או `payment in lieu`.
- אסטרטגיה ממונפת ללא מודל ריבית נראית טובה מדי.
- אסטרטגיית Long/Short יכולה להיות מוטה לכל כיוון.

לכן אסור להגן על התוצאות בטענה שהמנוע "מחמיר". המסקנה הנכונה היא שהמנוע כיום **מערבב הטיות פסימיות ואופטימיות**, והכיוון נטו אינו ידוע.

היעד הנכון הוא מנוע שמפריד בין:

1. מחירי מסחר אמיתיים;
2. תשואה כלכלית הכוללת דיבידנדים;
3. ספר חשבונות של מזומן, דיבידנדים לקבל, עסקאות שטרם נסלקו וריבית נצברת.

---

## מה קורה כיום ב־alpha_super

### שכבת הנתונים

הפונקציה המשותפת [`load_raw_prices`](../../data/norgate_loader.py) טוענת:

| שימוש | התאמת Norgate |
|---|---|
| נכסים נסחרים | `CAPITALSPECIAL` |
| בנצ'מרקים | `TOTALRETURN` |

`CAPITALSPECIAL` מטפל ב־splits, שינויי הון וחלוקות מיוחדות, אך אינו מכניס דיבידנד רגיל לתשואת המחיר. Norgate כן מספק עמודת `Dividend`, אבל מנוע [`Strategy.process_orders`](../../alpha/engine/strategy.py) אינו משתמש בה.

גם [`Strategy.update_metrics`](../../alpha/engine/strategy.py) מחשב את ה־NAV באמצעות:

```text
cash + shares × close
```

אין בו רכיב כללי של:

```text
dividend receivable
interest accrual
unsettled cash
```

### המשמעות לפי משפחת אסטרטגיות

| סוג אסטרטגיה | ההטיה הסבירה כיום | הסיבה |
|---|---|---|
| Long במניות או ETF | בדרך כלל פסימית | ירידת המחיר ב־ex-date נרשמת, הדיבידנד אינו נרשם |
| Short | בדרך כלל אופטימית | אין חיוב בגין הדיבידנד ואין `payment in lieu` |
| Long/Short | לא ידועה | ה־Long נפגע וה־Short נהנה; המשקל והעיתוי קובעים |
| TAA עם `TOTALRETURN` לסיגנל | הסיגנל קרוב יותר למציאות, ה־P&L עדיין חסר | חלק מאסטרטגיות TAA משתמשות ב־TR לסיגנל וב־CS לביצוע, ללא ספר דיבידנדים משותף |
| מינוף דרך יתרת מזומן שלילית | בדרך כלל אופטימית | אין ריבית מימון כללית במנוע |
| ETF ממונף כמו UPRO/TQQQ | עלות המינוף הפנימית כבר במחיר ה־ETF | אין צורך להוסיף ריבית חשבון אם החשיפה אינה יוצרת גם חוב מזומן |

יש סקריפטי מחקר שמזכים דיבידנד במפורש, למשל [`run_sector_dispersion_family_universe_study.py`](../../scripts/research/run_sector_dispersion_family_universe_study.py), אך זו אינה התנהגות של המנוע המשותף ולכן היא אינה חלה אוטומטית על אסטרטגיות BENCH.

### ה־All Weather הנוכחי

הווריאנט [`strategy_taa_levered_all_weather.py`](../../strategies/all_weather/strategy_taa_levered_all_weather.py) מוסיף ריבית קבועה של `2.4%` לשנה על מזומן שלילי, אך:

- הוא אינו רושם דיבידנדים של SPY, TLT או DBC;
- הריבית קבועה לאורך כל ההיסטוריה;
- החישוב הוא לפי `252` ימי מסחר ולא לפי ימי לוח;
- אין הבחנה בין מזומן נסחר לבין `settled cash`.

זה אינו מודל שמרני באופן ברור: השמטת דיבידנדים פסימית, אך ריבית של `2.4%` יכולה להיות אופטימית מאוד בתקופות ריבית גבוהה. לא ניתן לדעת את כיוון ההטיה הכולל בלי לבנות את שני הצדדים נכון.

---

## מה RealTest עושה

RealTest מספק נקודת ייחוס טובה יותר למנוע יומי:

- ברירת המחדל בייבוא Norgate היא `Adjustment: Capital`.
- הוא שומר מחירים היסטוריים במונחי **as traded**.
- התאמות ל־splits ולחישובי lookback מבוצעות בזמן החישוב.
- ברירת המחדל היא `IgnoreDividends: False`.
- Long מקבל דיבידנד ו־Short מחויב בדיבידנד.

מקורות: [Adjustment](https://mhptrading.com/docs/topics/idh-topic1390.htm), [IgnoreDividends](https://mhptrading.com/docs/topics/idh-topic10807.htm), [Dividend Handling](https://mhptrading.com/docs/topics/idh-topic1100.htm).

RealTest גם מזהיר ששימוש ב־`Adjustment: TotalReturn` אינו מומלץ לאסטרטגיית לייב: דיבידנדים מומרים ל־splits סינתטיים, ולכן כמות המניות בפקודת יציאה עלולה להיות שגויה. [OrdersUseQtyIn](https://mhptrading.com/docs/topics/idh-topic11173.htm)

עם זאת, RealTest אינו העתק מלא של IBKR:

- הוא מזכה את ה־equity בדיבידנד ב־ex-date;
- הוא אינו מחכה בהכרח ל־pay-date כדי להפוך את הזכאות למזומן;
- מודל הריבית שלו ניתן להגדרה באמצעות `MarginIntPct` ו־`RiskFreeRateSym`, אך אינו משחזר אוטומטית את כל מדרגות הריבית, הסגמנטים והסליקה של IBKR.

כלומר, RealTest טוב יותר מהמנוע הנוכחי בנושא דיבידנדים ומחירי as-traded, אך גם הוא משתמש בקירוב לצורך סימולציה.

---

## המודל שהמנוע שלנו צריך ליישם

```text
As-traded OHLC + corporate actions + dividend events
                         |
          +--------------+---------------+
          |                              |
          v                              v
   Signal return layer             Execution layer
   split-safe indicators           actual Open/Close
   economic total return           actual share quantity
          |                              |
          +--------------+---------------+
                         v
                  Broker-like ledger
          positions / settled cash / unsettled cash
          dividend receivable / accrued interest
                         |
                         v
                         NAV
```

### 1. תשואה לסיגנלים ולסיכון

לצורך momentum, volatility ו־covariance צריך להשתמש בתשואה כלכלית הכוללת דיבידנד:

$$
r^{total}_{i,t}
=
\frac{P_{i,t}+D_{i,t}}{P_{i,t-1}}-1
$$

בנתוני Norgate, `Dividend` נרשם ביום הזכאות — יום המסחר שלפני ה־ex-date. לכן הדיבידנד צריך להיכנס לתשואה שחוצה אל יום ה־ex-date, ולא כתוספת אקראית ביום הזכאות. [הגדרת Dividend של Norgate](https://norgatedata.com/data-content-tables.php)

סדרת `TOTALRETURN` יכולה לשמש בדיקת התאמה, אך לא מקור למחירי פקודות.

### 2. אירוע דיבידנד בספר החשבונות

אם הוחזקו \(q_i\) מניות בסוף יום הזכאות:

$$
Receivable_i = q_i \cdot D_i
$$

סדר האירועים הנכון:

1. בסוף יום הזכאות נשמרת הכמות הזכאית.
2. ב־ex-date נוצר `dividend receivable`; מחיר הנכס כבר נסחר ללא הדיבידנד.
3. ה־receivable נכלל ב־NAV, ולכן אין ירידת הון מלאכותית.
4. ב־pay-date ה־receivable הופך למזומן נטו ממס ועמלות.
5. הדיבידנד אינו מושקע אוטומטית; הוא נשאר במזומן עד האיזון הבא, אלא אם מדיניות הלייב מגדירה DRIP.

גם IBKR כולל `Dividend Accruals` בתוך NAV ומבטל אותם בעת תשלום המזומן. [IBKR Dividend Accruals](https://www.ibkrguides.com/reportingreference/reportguide/changeindividendaccruals_realized.htm)

### 3. ריבית ומזומן

ריבית על USD צריכה להיות מחושבת על `settled debit cash`, לפי ימי לוח:

$$
Interest_t
=
SettledDebitCash_t
\cdot Rate_t
\cdot
\frac{CalendarDays_t}{360}
$$

`Rate_t` צריך להיות ריבית הבסיס ההיסטורית בתוספת מרווח ומדרגת IBKR המתאימים לגודל החשבון. IBKR מחשב ריבית על יתרה מסולקת, צובר אותה מדי יום ומבצע posting חודשי. [IBKR Margin Interest](https://www.interactivebrokers.com/en/trading/margin-calculation-details.php)

נדרש גם לוח סליקה היסטורי:

- `T+2` לפני 28 במאי 2024;
- `T+1` החל מ־28 במאי 2024.  
  [SEC — מעבר ל־T+1](https://www.sec.gov/newsroom/press-releases/2024-62)

### 4. לייב מול בק־טסט

בבק־טסט המנוע צריך לשחזר את הספר לעיל. בלייב:

- Norgate הוא מקור הסיגנל והמחיר ההיסטורי;
- IBKR הוא מקור האמת לפוזיציות, `NetLiquidation`, מזומן, accruals, margin ו־fills;
- פקודת יום חדש חייבת להיבנות מהמצב האמיתי אצל הברוקר, לא ממצב מדומה שנשמר מקומית.

---

## סדר עדיפויות מומלץ

1. להוסיף למנוע אירועי דיבידנד אוניברסליים: זיכוי Long וחיוב Short.
2. להפריד בין `dividend receivable` לבין מזומן ששולם.
3. להוסיף מודל ריבית גלובלי לפי יתרה מסולקת, ימי לוח וריבית היסטורית.
4. להוסיף שכבת as-traded ושינויי כמות עקב splits, במקום להשתמש במחיר מותאם כאילו נסחר בפועל.
5. לבצע audit לכל משפחת אסטרטגיות ולסווג מחדש את התוצאות לאחר המעבר.
6. להשוות כל backtest ייצוגי מול RealTest ומול statement אמיתי של IBKR.

## מבחני קבלה למנוע

- ביום ex-date ללא תנועת שוק, NAV של Long אינו קופץ מטה.
- Short מחויב בדיוק בדיבידנד למניה.
- split משנה מחיר וכמות, אך אינו משנה NAV.
- pay-date משנה receivable ומזומן, אך לא NAV.
- ריבית סוף שבוע מחויבת לפי מספר ימי הלוח.
- קנייה או מכירה משנה ריבית רק במועד הסליקה המתאים.
- סדרת התשואה הכלכלית תואמת ל־Norgate `TOTALRETURN` בתוך סבילות מוגדרת.
- מצב הלייב נסגר מול IBKR: פוזיציות, מזומן, דיבידנדים, ריבית ו־NetLiq.

## פסק דין

המנוע הנוכחי טוב לבדיקת רעיונות ולשמירת timing של `T → Open T+1`, אבל הוא עדיין אינו מנוע חשבונאי שמדמה חשבון ברוקר מלא. חלק מתוצאות ה־Long נמוכות מדי, חלק מתוצאות ה־Short והמינוף גבוהות מדי, ולכן אין בסיס לומר שהתיק כולו "שמרני".

היעד אינו לעבור ל־`TOTALRETURN` בכל מקום. היעד הוא:

> **מחירי as-traded לביצוע, תשואה כלכלית לסיגנל, וספר חשבונות מפורש לדיבידנדים, סליקה וריבית.**

רק לאחר שהשכבה הזו קיימת ועוברת reconciliation מול RealTest ו־IBKR, ניתן להתייחס לתוצאות BENCH כקירוב אמין למסחר אמיתי.
