from sqlalchemy import Index, Column, Integer, String, Boolean, Float, JSON, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import declarative_base, mapped_column
from database.models.base import Base
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from geoalchemy2 import Geometry

class DendraStation(Base):
    __tablename__ = "station"
    __table_args__ = (
        Index("ix_dendra_st_id", "dendra_st_id", unique=True),
        {"schema": "dendra"},
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Dendra API fields
    dendra_st_id = mapped_column(String, unique=True)  # API's internal ID
    is_active = mapped_column(Boolean)
    is_stationary = mapped_column(Boolean)
    name = mapped_column(String)
    description = mapped_column(String)
    organization_id = mapped_column(String)
    slug = mapped_column(String)
    station_type = mapped_column(String)
    time_zone = mapped_column(String)
    utc_offset = mapped_column(Integer)
    external_refs = mapped_column(JSON)
    is_enabled = mapped_column(Boolean)
    is_geo_protected = mapped_column(Boolean)
    is_hidden = mapped_column(Boolean)
    state = mapped_column(String)
    access_levels = mapped_column(JSON)
    media = mapped_column(JSON)
    version_id = mapped_column(String)
    updated_at = mapped_column(DateTime)
    updated_by = mapped_column(String)
    created_at = mapped_column(DateTime)
    created_by = mapped_column(String)
    access_levels_resolved = mapped_column(JSON)
    general_config_resolved = mapped_column(JSON)
    organization_lookup = mapped_column(JSON)
    full_name = mapped_column(String)
    general_config = mapped_column(JSON)
    longitude = mapped_column(Float)
    latitude = mapped_column(Float)
    elevation = mapped_column(Float)
    geometry = Column(Geometry(geometry_type='POINT', srid=4326))
    

class DendraDatastream(Base):
    __tablename__ = "datastream"
    __table_args__ = (
        UniqueConstraint("dendra_ds_id", "station_id", name="uc_dendra_ds_id_station_id"),
        Index("ix_dendra_ds_id", "dendra_ds_id", unique=True),
        Index("ix_datastream_station", "station_id"),
        {"schema": "dendra"},
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Dendra API fields
    dendra_ds_id = Column(String, unique=True, nullable=False)
    station_id = Column(Integer, ForeignKey("dendra.station.id"))
    name = Column(String)
    description = Column(String)
    source_type = Column(String)
    state = Column(String)
    is_enabled = Column(Boolean)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)

    # extracted from terms
    variable = Column(String)
    medium = Column(String)
    unit = Column(String)

    # store the rest of the JSON for future use
    datastream_metadata = Column(JSON)



class DendraDatapoint(Base):
    __tablename__ = "datapoint"
    __table_args__ = (
        Index("ix_datapoint_datastream_ts", "datastream_id", "timestamp_utc"),
        {"schema": "dendra"},
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Dendra API fields
    datastream_id = mapped_column(Integer, ForeignKey("dendra.datastream.id", ondelete="CASCADE"), index=True)
    timestamp_utc = mapped_column(DateTime, index=True)
    value = mapped_column(Float)