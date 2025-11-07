from sqlalchemy import Index, Column, Integer, String, Boolean, Float, JSON, DateTime, ForeignKey, UniqueConstraint, ARRAY
from sqlalchemy.orm import declarative_base, mapped_column
from database.models.base import Base
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from geoalchemy2 import Geometry

class AnimlDeployment(Base):
    __tablename__ = "deployment"
    __table_args__ = (
        Index("ix_animl_dp_id", "animl_dp_id", unique=True),
        {"schema": "animl"},
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    animl_dp_id = mapped_column(String, unique=True, nullable=False)
    name = mapped_column(String)
    geometry = Column(Geometry(geometry_type='POINT', srid=4326))

    
class AnimlImage(Base):
    __tablename__ = "image"
    __table_args__ = (
        Index("ix_image_deployment", "deployment_id"),
        Index("ix_timestamp", "timestamp"),
        {"schema": "animl"},
    )

    id = mapped_column(Integer, primary_key=True, autoincrement=True)

    animl_image_id = mapped_column(String, unique=True, nullable=False)
    deployment_id = mapped_column(Integer, ForeignKey("animl.deployment.id"))
    timestamp = mapped_column(DateTime)
    labels = mapped_column(ARRAY(String), nullable=False)
    medium_url = mapped_column(String)
    small_url = mapped_column(String)
