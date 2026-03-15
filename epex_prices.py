"""ENTSO-E day-ahead price fetcher for EPEX NL spot prices."""

import logging
import os
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger(__name__)

ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
ENTSOE_TOKEN = os.environ.get("ENTSOE_TOKEN", "")
NL_DOMAIN = "10YNL----------L"
NS = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}

# Fixed cost components (EUR/kWh)
TAXES_PER_KWH = 0.125
PURCHASING_COSTS_PER_KWH = 0.02


class EPEXPriceTracker:
    def __init__(self):
        self._prices: dict[datetime, float] = {}  # hour_start → EUR/MWh
        self._lock = threading.Lock()

    def start(self):
        self._fetch_prices()
        threading.Thread(target=self._refresh_loop, daemon=True).start()

    def _refresh_loop(self):
        while True:
            time.sleep(3 * 3600)  # Refresh every 3 hours
            try:
                self._fetch_prices()
            except Exception:
                logger.exception("ENTSO-E price refresh failed")

    def refresh(self):
        """Public method to trigger a price refresh."""
        self._fetch_prices()

    def _fetch_prices(self):
        now = datetime.now(timezone.utc)
        # Fetch today and tomorrow
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=2)

        params = {
            "securityToken": ENTSOE_TOKEN,
            "documentType": "A44",
            "in_Domain": NL_DOMAIN,
            "out_Domain": NL_DOMAIN,
            "periodStart": start.strftime("%Y%m%d%H00"),
            "periodEnd": end.strftime("%Y%m%d%H00"),
        }

        try:
            r = httpx.get(ENTSOE_URL, params=params, timeout=30)
            r.raise_for_status()
            prices = self._parse_xml(r.text)
            with self._lock:
                self._prices.update(prices)
            logger.info("Fetched %d hourly prices from ENTSO-E", len(prices))
        except Exception:
            logger.exception("Failed to fetch ENTSO-E prices")

    def _parse_xml(self, xml_text: str) -> dict[datetime, float]:
        prices = {}
        root = ET.fromstring(xml_text)

        for ts in root.findall(".//ns:TimeSeries", NS):
            period = ts.find("ns:Period", NS)
            if period is None:
                continue

            start_text = period.findtext("ns:timeInterval/ns:start", namespaces=NS)
            if not start_text:
                continue

            start_dt = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
            resolution = period.findtext("ns:resolution", namespaces=NS)
            step = timedelta(hours=1) if resolution == "PT60M" else timedelta(minutes=15)

            for point in period.findall("ns:Point", NS):
                pos = int(point.findtext("ns:position", namespaces=NS)) - 1
                price_mwh = float(point.findtext("ns:price.amount", namespaces=NS))
                hour_start = start_dt + (step * pos)
                prices[hour_start] = price_mwh

        return prices

    def current_price_eur_per_kwh(self) -> float:
        now = datetime.now(timezone.utc)
        hour_start = now.replace(minute=0, second=0, microsecond=0)

        with self._lock:
            epex_mwh = self._prices.get(hour_start)

        if epex_mwh is not None:
            epex_kwh = epex_mwh / 1000  # EUR/MWh → EUR/kWh
        else:
            logger.warning("No EPEX price for %s, using 0", hour_start)
            epex_kwh = 0.0

        return TAXES_PER_KWH + epex_kwh + PURCHASING_COSTS_PER_KWH
