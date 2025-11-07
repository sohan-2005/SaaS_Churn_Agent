import json, os

NOTES_PATH = os.path.join('storage','notes.json')

def ingest_notes(notes):
    os.makedirs(os.path.dirname(NOTES_PATH), exist_ok=True)
    existing = []
    if os.path.exists(NOTES_PATH):
        existing = json.load(open(NOTES_PATH))
    existing.extend(notes)
    json.dump(existing, open(NOTES_PATH,'w'), indent=2)

def retrieve_relevant(customer_id, k=3):
    if not os.path.exists(NOTES_PATH):
        return []
    notes = json.load(open(NOTES_PATH))
    filtered = [n for n in notes if n.get('customer_id')==customer_id]
    return filtered[:k]