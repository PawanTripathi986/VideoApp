
import os
import uuid
from flask import Flask, request, jsonify, Response, send_from_directory
from script_generator import generate_video_from_characters, generate_individual_character_video, generate_video_conditional

# Alias fix for missing function name
from script_generator import generate_video_from_characters as generate_from_characters

from url_processor import process_url
from live_streamer import generate_live_frames
from ai_video_generator import AIVideoGenerator

from sqlalchemy import create_engine, Column, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database setup
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "cvapp_db")
DB_USER = os.getenv("DB_USER", "postgres")
import urllib.parse

DB_PASSWORD = os.getenv("DB_PASSWORD", "Pawan@321!")
DB_PASSWORD_ENCODED = urllib.parse.quote_plus(DB_PASSWORD)

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Character(Base):
    __tablename__ = "characters"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=True)
    color = Column(String, nullable=True)
    size = Column(String, nullable=True)
    script = Column(Text, nullable=True)

Base.metadata.create_all(bind=engine)

app = Flask(__name__)
OUTPUT_DIR = "output"

ai_video_generator = AIVideoGenerator()

# In-memory storage for characters
# Remove in-memory storage as we will use database
# stored_characters = {}

@app.route('/api/cartoon/script', methods=['POST'])
def cartoon_script():
    script = request.json.get("script")
    service = request.json.get("service", "default")
    if not script:
        return jsonify({"error": "No script provided"}), 400
    video_id = str(uuid.uuid4())
    if service == "default":
        path = generate_video_conditional(script, use_ai=False, video_id=video_id)
    else:
        try:
            path = ai_video_generator.generate_video(script, service)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
    return jsonify({"video_url": f"/output/{os.path.basename(path)}"})

@app.route('/api/cartoon/url', methods=['POST'])
def cartoon_url():
    url = request.json.get("video_url")
    if not url:
        return jsonify({"error": "No video_url provided"}), 400
    video_id = str(uuid.uuid4())
    path = process_url(url, video_id)
    return jsonify({"video_url": f"/output/{os.path.basename(path)}"})

@app.route('/api/cartoon/live', methods=['GET'])
def live_stream():
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/output/<filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/api/cartoon/characters', methods=['POST'])
def cartoon_characters():
    data = request.json
    # If data is a list, treat it as characters list directly
    if isinstance(data, list):
        characters = data
        use_ai = True
    else:
        characters = data.get("characters")
        use_ai = data.get("use_ai", True)
    if not characters:
        return jsonify({"error": "No characters provided"}), 400
    try:
        video_id = str(uuid.uuid4())
        video_path = generate_video_from_characters(characters, use_ai=use_ai, video_id=video_id)
        if not os.path.exists(video_path):
            return jsonify({"error": "Video generation failed, file not found"}), 500
        video_url = f"/output/{os.path.basename(video_path)}"
        return jsonify({"video_url": video_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cartoon/dynamic_character', methods=['POST'])
def dynamic_character():
    character = request.json
    if not character:
        return jsonify({"error": "No character data provided"}), 400
    try:
        video_id = str(uuid.uuid4())
        video_path = generate_individual_character_video(character, video_id)
        if not os.path.exists(video_path):
            return jsonify({"error": "Video generation failed, file not found"}), 500
        video_url = f"/output/{os.path.basename(video_path)}"
        return jsonify({"video_url": video_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cartoon/store_character', methods=['POST'])
def store_character():
    data = request.json
    if not data:
        return jsonify({"error": "No character data provided"}), 400

    session = SessionLocal()
    try:
        if isinstance(data, list):
            stored_ids = []
            for character in data:
                character_id = str(uuid.uuid4())
                db_character = Character(
                    id=character_id,
                    name=character.get("name"),
                    color=str(character.get("color")),
                    size=str(character.get("size")),
                    script=character.get("script")
                )
                session.add(db_character)
                stored_ids.append(character_id)
            session.commit()
            return jsonify({"message": "Characters stored", "character_ids": stored_ids})
        else:
            character_id = str(uuid.uuid4())
            db_character = Character(
                id=character_id,
                name=data.get("name"),
                color=str(data.get("color")),
                size=str(data.get("size")),
                script=data.get("script")
            )
            session.add(db_character)
            session.commit()
            return jsonify({"message": "Character stored", "character_id": character_id})
    except Exception as e:
        session.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

@app.route('/api/cartoon/generate_from_stored', methods=['POST'])
def generate_from_stored():
    import ast
    data = request.json
    character_ids = data.get("character_ids")
    use_ai = data.get("use_ai", True)
    if not character_ids or not isinstance(character_ids, list):
        return jsonify({"error": "character_ids must be a list of IDs"}), 400

    session = SessionLocal()
    try:
        characters = []
        for cid in character_ids:
            char = session.query(Character).filter(Character.id == cid).first()
            if not char:
                return jsonify({"error": f"Character ID {cid} not found"}), 404
            # Parse color and size back to original types
            try:
                color = ast.literal_eval(char.color) if char.color else None
            except Exception:
                color = None
            try:
                size = ast.literal_eval(char.size) if char.size else None
            except Exception:
                size = None
            characters.append({
                "name": char.name,
                "color": color,
                "size": size,
                "script": char.script
            })
        import uuid
        import os
        video_id = str(uuid.uuid4())
        video_path = generate_video_from_characters(characters, use_ai=use_ai, video_id=video_id)
        if not os.path.exists(video_path):
            return jsonify({"error": "Video generation failed, file not found"}), 500
        video_url = f"/output/{os.path.basename(video_path)}"
        return jsonify({"video_url": video_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

@app.route('/api/cartoon/list_characters', methods=['GET'])
def list_characters():
    session = SessionLocal()
    try:
        characters = session.query(Character).all()
        result = []
        for char in characters:
            result.append({
                "id": char.id,
                "name": char.name,
                "color": char.color,
                "size": char.size,
                "script": char.script
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        session.close()

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
