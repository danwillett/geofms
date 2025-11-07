from sqlalchemy.orm import sessionmaker

# create a session from postgres engine
def create_session(engine):
    session = sessionmaker()
    session.configure(bind=engine)
    s = session()
    return s