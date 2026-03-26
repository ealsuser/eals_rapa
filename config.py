from dataclasses import dataclass

@dataclass
class Paths:
    """Configuration settings for the application."""
    base: str = "/home/ubuntu/git/data-output/s3_rapa/"
    demographics: str = base + "demographics/RAPA EAP Demographics Data.csv"
    alsfrsr: str = base + "demographics/RAPA EAP ALSFRS Data.csv"
    roads: str = base + "demographics/RAPA EAP ROADS Data.csv"
    s3_bucket: str = 'eals-rapa-demographics-bucket'

@dataclass
class Aural:
    """Paths related to Aural Analytics data."""
    raw: str = Paths.base + "aural/"
    preprocessed: str = Paths.base + "preprocessed/" + "aural/"
    s3_bucket: str = 'eals-rapa-aural-bucket'
    

@dataclass
class Zephyrx:
    """Paths related to ZephyrX data."""
    raw: str = Paths.base + "zephyrx/tests/"
    preprocessed: str = Paths.base + "preprocessed/" + "zephyrx/"
    s3_bucket: str = 'eals-rapa-zephyrx-bucket'
    # aws s3 ls s3://eals-rapa-zephyrx-bucket