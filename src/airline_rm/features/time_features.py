"""Route-level and market-level aggregate features.

Since the dataset has no temporal columns (Year/Quarter), we compute
route-level statistical features as proxies for market dynamics.
"""

from __future__ import annotations

import polars as pl
from loguru import logger


def build_route_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add route-level aggregate features.

    Features computed per ODPairID:
    - route_avg_fare: average fare on this route
    - route_std_fare: fare variance on route
    - route_carrier_count: number of carriers on route
    - route_total_pax: total passengers on route
    - route_avg_market_share: average market share
    - route_competition_level: categorical competition indicator
    """
    route_stats = (
        df.group_by("ODPairID")
        .agg([
            pl.col("Average_Fare").mean().alias("route_avg_fare"),
            pl.col("Average_Fare").std().alias("route_std_fare"),
            pl.col("Carrier").n_unique().alias("route_carrier_count"),
            pl.col("Pax").sum().alias("route_total_pax"),
            pl.col("Market_share").mean().alias("route_avg_market_share"),
            pl.col("Market_HHI").first().alias("route_hhi"),
        ])
    )

    df = df.join(route_stats, on="ODPairID", how="left")

    # Competition level bins
    df = df.with_columns(
        pl.when(pl.col("route_carrier_count") == 1)
        .then(pl.lit(0))  # monopoly
        .when(pl.col("route_carrier_count") <= 3)
        .then(pl.lit(1))  # oligopoly
        .otherwise(pl.lit(2))  # competitive
        .alias("route_competition_level")
    )

    logger.info(f"Added {6} route-level features")
    return df


def build_carrier_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add carrier-level aggregate features.

    Features computed per Carrier:
    - carrier_avg_fare: carrier's average fare across all routes
    - carrier_route_count: number of routes the carrier operates
    - carrier_avg_distance: average distance flown by carrier
    - carrier_pax_share: carrier's share of total passengers
    """
    total_pax = df["Pax"].sum()

    carrier_stats = (
        df.group_by("Carrier")
        .agg([
            pl.col("Average_Fare").mean().alias("carrier_avg_fare"),
            pl.col("ODPairID").n_unique().alias("carrier_route_count"),
            pl.col("NonStopMiles").mean().alias("carrier_avg_distance"),
            (pl.col("CarrierPax").sum() / total_pax).alias("carrier_pax_share"),
        ])
    )

    df = df.join(carrier_stats, on="Carrier", how="left")
    logger.info("Added 4 carrier-level features")
    return df


def build_market_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add market-level features.

    - fare_per_mile: fare efficiency
    - circuity_premium: extra cost from circuity
    - is_hub_origin / is_hub_dest: frequency-based hub detection
    """
    df = df.with_columns([
        (pl.col("Average_Fare") / pl.col("NonStopMiles").clip(lower_bound=1))
        .alias("fare_per_mile"),
        (pl.col("Circuity") - 1.0).clip(lower_bound=0).alias("circuity_premium"),
    ])

    # Hub detection: airports appearing in top 5% by frequency
    for col, alias in [
        ("OriginAirportID", "is_hub_origin"),
        ("DestAirportID", "is_hub_dest"),
    ]:
        if col in df.columns:
            freq = df[col].value_counts()
            threshold = freq["count"].quantile(0.95)
            hub_ids = freq.filter(pl.col("count") >= threshold)[col]
            df = df.with_columns(
                pl.col(col).is_in(hub_ids).cast(pl.Int8).alias(alias)
            )

    logger.info("Added market-level features (fare_per_mile, circuity_premium, hub flags)")
    return df
