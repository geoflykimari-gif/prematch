import csv
import unicodedata
from collections import defaultdict

# ---------------------- File paths ----------------------
MASTER_CSV = "master_matches.csv"
UPCOMING_CSV = "upcoming_matches.csv"
MASTER_CLEANED = "master_cleaned.csv"
UPCOMING_CLEANED = "upcoming_cleaned.csv"
ALIAS_FILE = "team_aliases.csv"

# ---------------------- Helper Functions ----------------------
def normalize_name(name):
    """
    Normalize a team name: lowercase, remove accents, strip spaces.
    """
    if not name:
        return ""
    name = name.strip().lower()
    name = ''.join(c for c in unicodedata.normalize('NFD', name) if unicodedata.category(c) != 'Mn')
    name = ' '.join(name.split())
    return name

def load_teams(csv_files, team_cols=['home','away']):
    """
    Load unique team names from multiple CSVs.
    """
    teams = set()
    for file in csv_files:
        with open(file, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in team_cols:
                    if col in row and row[col].strip():
                        teams.add(row[col].strip())
    return teams

def build_alias_map(teams):
    """
    Build normalized -> canonical mapping.
    """
    alias_map = {}
    for t in teams:
        norm = normalize_name(t)
        if norm not in alias_map:
            alias_map[norm] = t
    return alias_map

def save_aliases(alias_map, file_path=ALIAS_FILE):
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['normalized_name','canonical_name'])
        for norm, canon in sorted(alias_map.items()):
            writer.writerow([norm, canon])

def clean_csv(input_file, output_file, alias_map, team_cols=['home','away']):
    """
    Produce a cleaned CSV with team names harmonized using alias_map.
    Keeps order and all other data intact.
    """
    with open(input_file, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Harmonize team names
    for row in rows:
        for col in team_cols:
            if col in row:
                norm = normalize_name(row[col])
                row[col] = alias_map.get(norm, row[col])

    # Write cleaned CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    # Step 1: Load all team names
    all_teams = load_teams([MASTER_CSV, UPCOMING_CSV])
    
    # Step 2: Build alias map
    alias_map = build_alias_map(all_teams)
    
    # Step 3: Save alias map
    save_aliases(alias_map)
    print(f"Generated alias map for {len(alias_map)} teams in {ALIAS_FILE}")
    
    # Step 4: Clean both CSVs
    clean_csv(MASTER_CSV, MASTER_CLEANED, alias_map)
    print(f"Cleaned master CSV saved as {MASTER_CLEANED}")
    
    clean_csv(UPCOMING_CSV, UPCOMING_CLEANED, alias_map)
    print(f"Cleaned upcoming CSV saved as {UPCOMING_CLEANED}")
