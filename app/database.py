from sqlalchemy import Column, Integer, String, Float, JSON,  DateTime, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

# Database URL
DATABASE_URL = f"sqlite:///{current_dir}/database.db"

# SQLAlchemy Setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Device 
class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String, index=True)
    country = Column(String)
    lat = Column(Float)
    lng = Column(Float)
    date_deployed = Column(DateTime)

    model_name = Column(String)
    model_checkpoint = Column(String)
    date_updated = Column(DateTime)

    audio_files = relationship("Audio", back_populates="device")

# Audio File 
class Audio(Base):
    __tablename__ = "audio"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    sample_rate = Column(Integer)
    date_recorded = Column(DateTime)

    device_id = Column(Integer, ForeignKey("devices.id"))

    device = relationship("Device", back_populates="audio_files")
    segments = relationship("Segment", back_populates="audio_file")

# Segment
class Segment(Base):
    __tablename__ = "segments"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=True)
    start_time = Column(Float)
    duration = Column(Float)
    uncertainty = Column(Float)
    energy = Column(JSON)
    date_processed = Column(DateTime)
    embedding_id = Column(Integer, index=True)

    # Human Annotations
    label = Column(String, index=True)
    notes = Column(String)

    audio_id = Column(Integer, ForeignKey("audio.id"))

    audio_file = relationship("Audio", back_populates="segments")
    predictions = relationship("Predictions", back_populates="segment")

# Prediction
class Predictions(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)

    # Model Predictions
    predicted_species = Column(String, index=True)
    confidence = Column(Float)

    segment_id = Column(Integer, ForeignKey("segments.id"))
    segment = relationship("Segment", back_populates="predictions")


Base.metadata.create_all(bind=engine)