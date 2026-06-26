"""A hand-labelled gold set of analyst recommendations for auditing the extractor.

~60 examples chosen to be REALISTIC and DELIBERATELY HARD, so the evaluation surfaces genuine
failure modes rather than a perfect diagonal: upgrade/downgrade "from X to Y" transitions (which a
keyword parser tends to read backwards), unsupported rating synonyms (peer perform / sector weight),
targets stated as ranges / "from-to" / without a $ sign, current-price amounts that look like
targets, multi-ticker text, firm-as-subject, missing fields, and varied date formats.

Each example carries the ground-truth fields a human would extract; ``None`` means "absent".
Ratings are the five canonical labels (Buy / Overweight / Hold / Underweight / Sell).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GoldExample:
    text: str
    ticker: Optional[str]
    rating: Optional[str]
    target_price: Optional[float]
    analyst: Optional[str] = None
    firm: Optional[str] = None
    date: Optional[str] = None        # ISO YYYY-MM-DD or None
    note: str = ""                    # what makes this case interesting


GOLD: list[GoldExample] = [
    # --- clean Buy ---
    GoldExample("Goldman Sachs reiterates Buy on Apple (NASDAQ: AAPL), price target $260.",
                "AAPL", "Buy", 260.0, None, "Goldman Sachs", None, "clean"),
    GoldExample("We initiate $MSFT at Buy with a $520 target.", "MSFT", "Buy", 520.0, None, None, None, "clean"),
    GoldExample("Morgan Stanley analyst Katy Huberty keeps a Buy on Nvidia, target $180, May 5, 2025.",
                "NVDA", "Buy", 180.0, "Katy Huberty", "Morgan Stanley", "2025-05-05", "clean+date+analyst"),
    GoldExample("Strong Buy: Coinbase ($COIN), price target $400.", "COIN", "Buy", 400.0, None, None, None, "clean"),

    # --- Overweight / Outperform / Accumulate ---
    GoldExample("Barclays raises Amazon to Overweight, target $250.", "AMZN", "Overweight", 250.0, None, "Barclays", None, "clean"),
    GoldExample("RBC analyst Tom Narayan rates Ford Outperform, PT $14.", "F", "Overweight", 14.0, "Tom Narayan", "RBC", None, "synonym outperform"),
    GoldExample("Citi keeps Accumulate on Exxon, target $130.", "XOM", "Overweight", 130.0, None, "Citi", None, "synonym accumulate"),
    GoldExample("Wedbush: Outperform on Tesla (TSLA), price target $350.", "TSLA", "Overweight", 350.0, None, "Wedbush", None, "synonym"),

    # --- Hold / Neutral / Equal-weight / Market perform ---
    GoldExample("UBS maintains Neutral on Meta (META), target $600.", "META", "Hold", 600.0, None, "UBS", None, "synonym neutral"),
    GoldExample("JPMorgan rates Disney Hold with a $250 target.", "DIS", "Hold", 250.0, None, "JPMorgan", None, "clean"),
    GoldExample("Stifel keeps Equal-Weight on Intel (INTC), price target $22.", "INTC", "Hold", 22.0, None, "Stifel", None, "synonym equal-weight"),
    GoldExample("KeyBanc has a Market Perform rating on Micron, target $90.", "MU", "Hold", 90.0, None, "KeyBanc", None, "synonym market perform"),

    # --- Underweight / Underperform / Reduce ---
    GoldExample("Morgan Stanley cuts Palantir to Underweight, target $20.", "PLTR", "Underweight", 20.0, None, "Morgan Stanley", None, "clean"),
    GoldExample("Jefferies rates Snowflake Underperform, PT $110.", "SNOW", "Underweight", 110.0, None, "Jefferies", None, "synonym underperform"),
    GoldExample("We reduce Boeing to a $180 target, Reduce rating.", "BA", "Underweight", 180.0, None, None, None, "synonym reduce"),
    GoldExample("Bernstein keeps Underweight on Netflix (NFLX), target $500.", "NFLX", "Underweight", 500.0, None, "Bernstein", None, "clean"),

    # --- Sell ---
    GoldExample("Goldman Sachs downgrades Shopify to Sell, $60 target.", "SHOP", "Sell", 60.0, None, "Goldman Sachs", None, "clean"),
    GoldExample("Sell rating on Palo Alto Networks, price target $150.", "PANW", "Sell", 150.0, None, None, None, "name->ticker"),
    GoldExample("$NKE Sell — target $60, demand softening.", "NKE", "Sell", 60.0, None, None, None, "cashtag"),

    # --- upgrade / downgrade transitions (the parser must read the NEW rating, after 'to') ---
    GoldExample("Citi upgrades Salesforce from Hold to Buy, target $400.", "CRM", "Buy", 400.0, None, "Citi", None, "transition->Buy"),
    GoldExample("Morgan Stanley downgrades AMD from Buy to Hold, $190 target.", "AMD", "Hold", 190.0, None, "Morgan Stanley", None, "transition->Hold"),
    GoldExample("Wells Fargo upgrades Uber from Underweight to Overweight, target $90.", "UBER", "Overweight", 90.0, None, "Wells Fargo", None, "transition->OW"),
    GoldExample("Oppenheimer downgrades Nvidia from Buy to Hold but keeps a $170 target.", "NVDA", "Hold", 170.0, None, "Oppenheimer", None, "transition->Hold"),
    GoldExample("Bank of America upgrades Netflix from Underperform to Neutral, target $520.", "NFLX", "Hold", 520.0, None, "Bank of America", None, "transition->Hold"),
    GoldExample("Raymond James cuts Coinbase from Buy to Underperform, $150 target.", "COIN", "Underweight", 150.0, None, "Raymond James", None, "transition->UW"),
    GoldExample("Mizuho upgrades Intel from Sell to Hold, target $25.", "INTC", "Hold", 25.0, None, "Mizuho", None, "transition->Hold"),
    GoldExample("Deutsche Bank downgrades Apple from Overweight to Hold, $230 target.", "AAPL", "Hold", 230.0, None, "Deutsche Bank", None, "transition->Hold"),
    GoldExample("Citi: Buy, Hold, or Sell? We say Buy. Apple $260 target.", "AAPL", "Buy", 260.0, None, "Citi", None, "multi-rating-words"),

    # --- unsupported Hold synonyms ---
    GoldExample("Cowen rates Disney Peer Perform, target $250.", "DIS", "Hold", 250.0, None, "Cowen", None, "synonym peer perform"),
    GoldExample("KeyBanc has a Sector Weight rating on Micron.", "MU", "Hold", None, None, "KeyBanc", None, "synonym sector weight + no target"),
    GoldExample("Wells Fargo: Market Weight on Boeing, $200 target.", "BA", "Hold", 200.0, None, "Wells Fargo", None, "synonym market weight"),
    GoldExample("We rate AMD In-Line, target $50.", "AMD", "Hold", 50.0, None, None, None, "synonym in-line"),

    # --- missing rating (target only) ---
    GoldExample("Apple price target raised to $260 at Goldman Sachs.", "AAPL", None, 260.0, None, "Goldman Sachs", None, "no rating"),
    GoldExample("$TSLA price target $350.", "TSLA", None, 350.0, None, None, None, "no rating"),
    GoldExample("Nvidia target lifted to $180.", "NVDA", None, 180.0, None, None, None, "no rating"),

    # --- missing target (rating only) — watch for hallucinated targets from stray prices ---
    GoldExample("Citi keeps Buy on Amazon.", "AMZN", "Buy", None, None, "Citi", None, "no target"),
    GoldExample("We rate Apple a Buy; shares last traded at $195.", "AAPL", "Buy", None, None, None, None, "price is NOT a target"),
    GoldExample("Hold on Disney.", "DIS", "Hold", None, None, None, None, "no target"),

    # --- missing ticker ---
    GoldExample("We upgrade to Buy with a $300 target.", None, "Buy", 300.0, None, None, None, "no ticker"),
    GoldExample("Strong buy here, price target $50.", None, "Buy", 50.0, None, None, None, "no ticker"),

    # --- tricky tickers ---
    GoldExample("Bank of America upgraded shares of AMD to Buy, target $190.", "AMD", "Buy", 190.0, None, "Bank of America", None, "firm-as-noise"),
    GoldExample("Pair trade: long $NVDA, short $AMD. Buy NVDA, $180 target.", "NVDA", "Buy", 180.0, None, None, None, "multi-ticker (subject NVDA)"),
    GoldExample("Alphabet (GOOGL) gets a Buy at Wells Fargo, $220 target.", "GOOGL", "Buy", 220.0, None, "Wells Fargo", None, "name+paren"),
    GoldExample("Palo Alto Networks (PANW) raised to Buy, target $400.", "PANW", "Buy", 400.0, None, None, None, "long name"),

    # --- tricky targets ---
    GoldExample("Morgan Stanley sets a price target range of $300-$320 on Tesla, Overweight.", "TSLA", "Overweight", 300.0, None, "Morgan Stanley", None, "range (low)"),
    GoldExample("We raise our Apple target to $260 from $240.", "AAPL", None, 260.0, None, None, None, "from-to (new=260)"),
    GoldExample("Target of 300 (no dollar sign) on Nvidia, Buy.", "NVDA", "Buy", 300.0, None, None, None, "no $ sign"),
    GoldExample("Outperform, $14.50 price target on Ford.", "F", "Overweight", 14.5, None, None, None, "decimal target"),

    # --- dates ---
    GoldExample("On 2025-03-14, Citi reiterated Buy on Apple, $260 target.", "AAPL", "Buy", 260.0, None, "Citi", "2025-03-14", "iso date"),
    GoldExample("March 14, 2025 - UBS keeps Neutral on Meta, $600 PT.", "META", "Hold", 600.0, None, "UBS", "2025-03-14", "long date"),
    GoldExample("3/14/2025: Sell Coinbase, target $150.", "COIN", "Sell", 150.0, None, None, "2025-03-14", "slash date"),
    GoldExample("14 Mar 2025, Buy Nvidia, $180 target.", "NVDA", "Buy", 180.0, None, None, "2025-03-14", "dmy date"),
    GoldExample("Updated Q1 2025: Overweight Amazon, $250.", "AMZN", "Overweight", 250.0, None, None, None, "quarter, no parseable date"),

    # --- analyst name formats ---
    GoldExample("Wedbush analyst Dan Ives reiterates Buy on Apple, $300 target.", "AAPL", "Buy", 300.0, "Dan Ives", "Wedbush", None, "analyst X"),
    GoldExample("Gene Munger of Loop Capital rates Tesla Buy, $350.", "TSLA", "Buy", 350.0, "Gene Munger", "Loop Capital", None, "X of FIRM"),
    GoldExample("Per Mark Mahaney, analyst, Buy Netflix, $700 target.", "NFLX", "Buy", 700.0, "Mark Mahaney", None, None, "X, analyst"),
    GoldExample("Analyst Toni Sacconaghi keeps Hold on Apple, $230.", "AAPL", "Hold", 230.0, "Toni Sacconaghi", None, None, "analyst X"),

    # --- misc ---
    GoldExample("Downgrade to Sell. $TSLA target $200.", "TSLA", "Sell", 200.0, None, None, None, "clean"),
    GoldExample("Reiterate Overweight, raise PT to $175 from $150, Nvidia.", "NVDA", "Overweight", 175.0, None, None, None, "PT from-to"),
    GoldExample("We see 30% upside; Buy AAPL.", "AAPL", "Buy", None, None, None, None, "% not a target"),

    # --- adversarial: known limits of a keyword heuristic (the LLM path handles these) ---
    GoldExample("We rate it a Buy, definitely not a Sell.", None, "Buy", None, None, None, None,
                "ADVERSARIAL: negation — heuristic takes the last word (Sell)"),
    GoldExample("Bull case implies a $400 target; our base-case target is $300, Overweight Tesla.",
                "TSLA", "Overweight", 300.0, None, None, None,
                "ADVERSARIAL: scenario targets — heuristic takes the first ($400)"),
    GoldExample("Reiterate Buy on Apple, $260 target. (Prior rating: Sell.)", "AAPL", "Buy", 260.0,
                None, None, None, "ADVERSARIAL: parenthetical prior rating — heuristic takes the last (Sell)"),
    GoldExample("Buy AAPL ($260 target); we'd Sell MSFT here.", "AAPL", "Buy", 260.0, None, None, None,
                "ADVERSARIAL: two stocks, opposite calls — heuristic takes the last rating (Sell)"),
]
