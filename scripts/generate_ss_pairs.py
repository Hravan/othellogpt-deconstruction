"""
scripts/generate_ss_pairs.py

Generate semantic equivalence groups for the Semantic Sensitivity (SS) metric.

Each group contains 2-3 questions that are semantically identical.
A model with stable semantic representations should assign nearly the same
output distribution to each question in a group.

Each group has an "answer_type" field:
  "yes_no"  — expected answer is "yes" or "no"
  "word"    — expected answer is a specific word or short phrase

Categories
----------
capital_word_order     "Is X the capital of Y?" ↔ "Is the capital of Y X?"  [yes + no]
arithmetic_order       "Is a+b=c?" ↔ "Is b+a=c?" ↔ "Is c the sum of a and b?"  [yes + no]
comparison_symmetric   "Is a > b?" ↔ "Is b < a?" ↔ "Is a larger than b?"  [yes + no]
active_passive         "Did X invent Y?" ↔ "Was Y invented by X?"  [yes]
geographic_containment "Is X in Y?" ↔ "Is X located in Y?" ↔ "Does Y contain X?"  [yes]
classification         "Is X a Y?" ↔ "Is X a type of Y?" ↔ "Would you classify X as a Y?"  [yes]
unit_equivalence       "Is 1 km equal to 1000 m?" ↔ "Does 1 km equal 1000 m?"  [yes]
chemical_formula       "Is the formula for X Y?" ↔ "Is Y the formula for X?"  [yes]
capital_retrieval      "What is the capital of X?" ↔ "Which city is the capital of X?"  [word]
arithmetic_result      "What is a plus b?" ↔ "What is the sum of a and b?"  [word]
"""

import json
import random
from pathlib import Path

SEED = 42
OUTPUT_PATH = Path("data/ss_pairs.json")

rng = random.Random(SEED)
groups: list[dict] = []


# ---------------------------------------------------------------------------
# 1. Capital cities — word order  (100 groups × 3 questions)
# ---------------------------------------------------------------------------

CAPITALS = [
    # Europe
    ("Albania", "Tirana"), ("Austria", "Vienna"), ("Belgium", "Brussels"),
    ("Bulgaria", "Sofia"), ("Croatia", "Zagreb"), ("Cyprus", "Nicosia"),
    ("Czech Republic", "Prague"), ("Denmark", "Copenhagen"),
    ("Estonia", "Tallinn"), ("Finland", "Helsinki"), ("France", "Paris"),
    ("Germany", "Berlin"), ("Greece", "Athens"), ("Hungary", "Budapest"),
    ("Iceland", "Reykjavik"), ("Ireland", "Dublin"), ("Italy", "Rome"),
    ("Latvia", "Riga"), ("Lithuania", "Vilnius"), ("Luxembourg", "Luxembourg City"),
    ("Malta", "Valletta"), ("Netherlands", "Amsterdam"), ("Norway", "Oslo"),
    ("Poland", "Warsaw"), ("Portugal", "Lisbon"), ("Romania", "Bucharest"),
    ("Serbia", "Belgrade"), ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"),
    ("Spain", "Madrid"), ("Sweden", "Stockholm"), ("Switzerland", "Bern"),
    ("United Kingdom", "London"), ("Ukraine", "Kyiv"), ("Belarus", "Minsk"),
    # Americas
    ("Argentina", "Buenos Aires"), ("Brazil", "Brasília"), ("Canada", "Ottawa"),
    ("Chile", "Santiago"), ("Colombia", "Bogotá"), ("Cuba", "Havana"),
    ("Ecuador", "Quito"), ("Jamaica", "Kingston"), ("Mexico", "Mexico City"),
    ("Panama", "Panama City"), ("Peru", "Lima"), ("Uruguay", "Montevideo"),
    ("Venezuela", "Caracas"), ("United States", "Washington, D.C."),
    ("Guatemala", "Guatemala City"),
    # Asia
    ("Armenia", "Yerevan"), ("Azerbaijan", "Baku"), ("Bangladesh", "Dhaka"),
    ("Cambodia", "Phnom Penh"), ("China", "Beijing"), ("Georgia", "Tbilisi"),
    ("India", "New Delhi"), ("Indonesia", "Jakarta"), ("Iran", "Tehran"),
    ("Iraq", "Baghdad"), ("Japan", "Tokyo"), ("Jordan", "Amman"),
    ("Kazakhstan", "Astana"), ("Kuwait", "Kuwait City"), ("Lebanon", "Beirut"),
    ("Malaysia", "Kuala Lumpur"), ("Mongolia", "Ulaanbaatar"),
    ("Nepal", "Kathmandu"), ("Pakistan", "Islamabad"), ("Philippines", "Manila"),
    ("Russia", "Moscow"), ("Saudi Arabia", "Riyadh"), ("South Korea", "Seoul"),
    ("Taiwan", "Taipei"), ("Thailand", "Bangkok"), ("Turkey", "Ankara"),
    ("Vietnam", "Hanoi"), ("United Arab Emirates", "Abu Dhabi"),
    ("Singapore", "Singapore"), ("Syria", "Damascus"),
    # Africa & Oceania
    ("Algeria", "Algiers"), ("Angola", "Luanda"), ("Australia", "Canberra"),
    ("Cameroon", "Yaoundé"), ("Egypt", "Cairo"), ("Ethiopia", "Addis Ababa"),
    ("Ghana", "Accra"), ("Kenya", "Nairobi"), ("Libya", "Tripoli"),
    ("Morocco", "Rabat"), ("Mozambique", "Maputo"), ("New Zealand", "Wellington"),
    ("Nigeria", "Abuja"), ("Sudan", "Khartoum"), ("Tanzania", "Dodoma"),
    ("Tunisia", "Tunis"), ("Uganda", "Kampala"), ("Zimbabwe", "Harare"),
    ("Bhutan", "Thimphu"), ("Somalia", "Mogadishu"),
]

for country, capital in CAPITALS:
    groups.append({
        "category": "capital_word_order",
        "answer_type": "yes_no",
        "questions": [
            f"Is {capital} the capital of {country}?",
            f"Is the capital of {country} {capital}?",
            f"Does {country} have {capital} as its capital?",
        ],
        "expected": "yes",
    })

# Wrong capitals — no answers
shuffled_capitals = CAPITALS[:]
rng.shuffle(shuffled_capitals)
wrong_capital_pairs: list[tuple[str, str, str]] = []
for country, correct_capital in CAPITALS:
    for _, wrong_capital in shuffled_capitals:
        if wrong_capital != correct_capital:
            wrong_capital_pairs.append((country, correct_capital, wrong_capital))
            break

for country, _correct_capital, wrong_capital in wrong_capital_pairs[:100]:
    groups.append({
        "category": "capital_word_order",
        "answer_type": "yes_no",
        "questions": [
            f"Is {wrong_capital} the capital of {country}?",
            f"Is the capital of {country} {wrong_capital}?",
            f"Does {country} have {wrong_capital} as its capital?",
        ],
        "expected": "no",
    })


# ---------------------------------------------------------------------------
# 2. Arithmetic commutativity  (300 groups × 3 questions)
# ---------------------------------------------------------------------------

seen_sums: set[tuple] = set()
while len(seen_sums) < 300:
    a = rng.randint(2, 60)
    b = rng.randint(2, 60)
    key = (min(a, b), max(a, b))
    if key in seen_sums:
        continue
    seen_sums.add(key)
    c = a + b
    groups.append({
        "category": "arithmetic_order",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} plus {b} equal to {c}?",
            f"Does {b} plus {a} equal {c}?",
            f"Is {c} the sum of {a} and {b}?",
        ],
        "expected": "yes",
    })

# Wrong sums — no answers
seen_no_sums: set[tuple] = set()
while len(seen_no_sums) < 100:
    a = rng.randint(2, 60)
    b = rng.randint(2, 60)
    key = (min(a, b), max(a, b))
    if key in seen_no_sums or key in seen_sums:
        continue
    seen_no_sums.add(key)
    c = a + b
    wrong_c = c + rng.choice([-2, -1, 1, 2])
    groups.append({
        "category": "arithmetic_order",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} plus {b} equal to {wrong_c}?",
            f"Does {b} plus {a} equal {wrong_c}?",
            f"Is {wrong_c} the sum of {a} and {b}?",
        ],
        "expected": "no",
    })


# ---------------------------------------------------------------------------
# 3. Comparison symmetry  (200 groups × 3 questions)
# ---------------------------------------------------------------------------

seen_pairs: set[tuple] = set()
while len(seen_pairs) < 200:
    a = rng.randint(2, 300)
    b = rng.randint(2, 300)
    if a == b:
        continue
    large, small = max(a, b), min(a, b)
    key = (large, small)
    if key in seen_pairs:
        continue
    seen_pairs.add(key)
    groups.append({
        "category": "comparison_symmetric",
        "answer_type": "yes_no",
        "questions": [
            f"Is {large} greater than {small}?",
            f"Is {small} less than {large}?",
            f"Is {large} larger than {small}?",
        ],
        "expected": "yes",
    })

# Inverted comparisons — no answers
seen_no_pairs: set[tuple] = set()
while len(seen_no_pairs) < 100:
    a = rng.randint(2, 300)
    b = rng.randint(2, 300)
    if a == b:
        continue
    large, small = max(a, b), min(a, b)
    key = (large, small)
    if key in seen_no_pairs or key in seen_pairs:
        continue
    seen_no_pairs.add(key)
    groups.append({
        "category": "comparison_symmetric",
        "answer_type": "yes_no",
        "questions": [
            f"Is {small} greater than {large}?",
            f"Is {large} less than {small}?",
            f"Is {small} larger than {large}?",
        ],
        "expected": "no",
    })


# ---------------------------------------------------------------------------
# 4. Active / passive voice  (50 groups × 2 questions)
# ---------------------------------------------------------------------------

INVENTIONS = [
    ("Alexander Fleming", "discovered", "penicillin"),
    ("Tim Berners-Lee", "invented", "the World Wide Web"),
    ("Johannes Gutenberg", "invented", "the printing press"),
    ("Alexander Graham Bell", "invented", "the telephone"),
    ("Thomas Edison", "invented", "the phonograph"),
    ("Karl Benz", "invented", "the automobile"),
    ("Alfred Nobel", "invented", "dynamite"),
    ("Louis Pasteur", "developed", "pasteurization"),
    ("Guglielmo Marconi", "invented", "the radio"),
    ("Charles Darwin", "proposed", "the theory of evolution"),
    ("Isaac Newton", "formulated", "the law of universal gravitation"),
    ("Marie Curie", "discovered", "polonium"),
    ("Nikola Tesla", "invented", "the alternating current motor"),
    ("Henry Ford", "developed", "the moving assembly line"),
    ("Dmitri Mendeleev", "created", "the periodic table of elements"),
    ("James Watt", "improved", "the steam engine"),
    ("Michael Faraday", "discovered", "electromagnetic induction"),
    ("Louis Daguerre", "invented", "the daguerreotype"),
    ("Samuel Morse", "invented", "Morse code"),
    ("Blaise Pascal", "invented", "the mechanical calculator"),
    ("George Eastman", "invented", "roll film photography"),
    ("John Logie Baird", "demonstrated", "the first working television"),
    ("Rudolf Diesel", "invented", "the diesel engine"),
    ("Antonie van Leeuwenhoek", "invented", "the first practical microscope"),
    ("Christian Huygens", "invented", "the pendulum clock"),
    ("Humphry Davy", "invented", "the arc lamp"),
    ("William Henry Perkin", "synthesized", "the first synthetic dye"),
    ("Hans Christian Ørsted", "discovered", "electromagnetism"),
    ("Niels Bohr", "proposed", "the Bohr model of the atom"),
    ("Ernest Rutherford", "discovered", "the atomic nucleus"),
    ("Max Planck", "proposed", "quantum theory"),
    ("Werner Heisenberg", "formulated", "the uncertainty principle"),
    ("Enrico Fermi", "built", "the first nuclear reactor"),
    ("Linus Pauling", "discovered", "the nature of the chemical bond"),
    ("Evangelista Torricelli", "invented", "the barometer"),
    ("Anders Celsius", "invented", "the Celsius temperature scale"),
    ("Georg Simon Ohm", "formulated", "Ohm's law"),
    ("André-Marie Ampère", "formulated", "the theory of electromagnetism"),
    ("Otto von Guericke", "invented", "the vacuum pump"),
    ("Charles Babbage", "designed", "the first mechanical computer"),
    ("Ada Lovelace", "wrote", "the first computer program"),
    ("Alan Turing", "proposed", "the Turing machine"),
    ("Tim Berners-Lee", "created", "the HTTP protocol"),
    ("Vint Cerf", "co-developed", "the TCP/IP protocol"),
    ("Dennis Ritchie", "created", "the C programming language"),
    ("Linus Torvalds", "created", "the Linux kernel"),
    ("James Clerk Maxwell", "formulated", "the equations of electromagnetism"),
    ("Antoine Lavoisier", "discovered", "the law of conservation of mass"),
    ("Robert Boyle", "formulated", "Boyle's law of gas pressure"),
    ("Daniel Gabriel Fahrenheit", "invented", "the mercury thermometer"),
]

VERB_PAST_PARTICIPLE = {
    "invented": "invented", "discovered": "discovered", "proposed": "proposed",
    "formulated": "formulated", "developed": "developed", "created": "created",
    "improved": "improved", "designed": "designed", "demonstrated": "demonstrated",
    "synthesized": "synthesized", "built": "built", "wrote": "written",
    "co-developed": "co-developed",
}

for person, verb, thing in INVENTIONS:
    past_part = VERB_PAST_PARTICIPLE.get(verb, verb + "d")
    groups.append({
        "category": "active_passive",
        "answer_type": "yes_no",
        "questions": [
            f"Did {person} {verb} {thing}?",
            f"Was {thing} {past_part} by {person}?",
        ],
        "expected": "yes",
    })


# ---------------------------------------------------------------------------
# 5. Geographic containment  (100 groups × 3 questions)
# ---------------------------------------------------------------------------

LOCATIONS = [
    ("Paris", "France"), ("London", "the United Kingdom"), ("Tokyo", "Japan"),
    ("New York City", "the United States"), ("Sydney", "Australia"),
    ("Mumbai", "India"), ("São Paulo", "Brazil"), ("Berlin", "Germany"),
    ("Rome", "Italy"), ("Moscow", "Russia"), ("Cairo", "Egypt"),
    ("Beijing", "China"), ("Toronto", "Canada"), ("Madrid", "Spain"),
    ("Amsterdam", "the Netherlands"), ("Seoul", "South Korea"),
    ("Istanbul", "Turkey"), ("Buenos Aires", "Argentina"),
    ("Nairobi", "Kenya"), ("Bangkok", "Thailand"), ("Mexico City", "Mexico"),
    ("Lagos", "Nigeria"), ("Vienna", "Austria"), ("Warsaw", "Poland"),
    ("Athens", "Greece"), ("Prague", "the Czech Republic"),
    ("Stockholm", "Sweden"), ("Oslo", "Norway"), ("Helsinki", "Finland"),
    ("Lisbon", "Portugal"), ("Brussels", "Belgium"), ("Dublin", "Ireland"),
    ("Budapest", "Hungary"), ("Bucharest", "Romania"), ("Kyiv", "Ukraine"),
    ("Tehran", "Iran"), ("Riyadh", "Saudi Arabia"),
    ("Kuala Lumpur", "Malaysia"), ("Jakarta", "Indonesia"),
    ("Manila", "the Philippines"), ("Hanoi", "Vietnam"),
    ("Dhaka", "Bangladesh"), ("Karachi", "Pakistan"),
    ("Casablanca", "Morocco"), ("Accra", "Ghana"),
    ("Addis Ababa", "Ethiopia"), ("Kampala", "Uganda"),
    ("Dakar", "Senegal"), ("Havana", "Cuba"),
    ("Cape Town", "South Africa"), ("Marrakech", "Morocco"),
    ("Tunis", "Tunisia"), ("Alexandria", "Egypt"),
    ("Kathmandu", "Nepal"), ("Colombo", "Sri Lanka"),
    ("Phnom Penh", "Cambodia"), ("Taipei", "Taiwan"),
    ("Hong Kong", "China"), ("Ulaanbaatar", "Mongolia"),
    ("Baku", "Azerbaijan"), ("Tbilisi", "Georgia"),
    ("Yerevan", "Armenia"), ("Beirut", "Lebanon"),
    ("Amman", "Jordan"), ("Baghdad", "Iraq"),
    ("Dubai", "the United Arab Emirates"), ("Doha", "Qatar"),
    ("Kabul", "Afghanistan"), ("Delhi", "India"),
    ("Lahore", "Pakistan"), ("Ho Chi Minh City", "Vietnam"),
    ("Singapore", "Southeast Asia"), ("Johannesburg", "South Africa"),
    ("Khartoum", "Sudan"), ("Dar es Salaam", "Tanzania"),
    ("Yangon", "Myanmar"), ("Tashkent", "Uzbekistan"),
    ("Almaty", "Kazakhstan"), ("Kuwait City", "Kuwait"),
    ("Muscat", "Oman"), ("Islamabad", "Pakistan"),
    ("Chittagong", "Bangladesh"), ("Cebu", "the Philippines"),
    ("Surabaya", "Indonesia"), ("Penang", "Malaysia"),
    ("Chiang Mai", "Thailand"), ("Da Nang", "Vietnam"),
    ("Valencia", "Spain"), ("Seville", "Spain"),
    ("Lyon", "France"), ("Marseille", "France"),
    ("Munich", "Germany"), ("Hamburg", "Germany"),
    ("Milan", "Italy"), ("Naples", "Italy"),
    ("Kraków", "Poland"), ("Gdańsk", "Poland"),
    ("Łódź", "Poland"), ("Bratislava", "Slovakia"),
    ("Tallinn", "Estonia"), ("Vilnius", "Lithuania"),
    ("Riga", "Latvia"), ("Sofia", "Bulgaria"),
]

for place, container in LOCATIONS[:100]:
    groups.append({
        "category": "geographic_containment",
        "answer_type": "yes_no",
        "questions": [
            f"Is {place} in {container}?",
            f"Is {place} located in {container}?",
            f"Does {container} contain the city of {place}?",
        ],
        "expected": "yes",
    })


# ---------------------------------------------------------------------------
# 6. Classification  (100 groups × 3 questions)
# ---------------------------------------------------------------------------

# (thing, indefinite_article + category, noun_category)
CLASSIFICATIONS = [
    ("a whale", "a mammal", "mammals"),
    ("a bat", "a mammal", "mammals"),
    ("a dolphin", "a mammal", "mammals"),
    ("an eagle", "a bird", "birds"),
    ("a penguin", "a bird", "birds"),
    ("a salmon", "a fish", "fish"),
    ("a shark", "a fish", "fish"),
    ("a frog", "an amphibian", "amphibians"),
    ("a toad", "an amphibian", "amphibians"),
    ("a cobra", "a reptile", "reptiles"),
    ("a crocodile", "a reptile", "reptiles"),
    ("a ladybug", "an insect", "insects"),
    ("a butterfly", "an insect", "insects"),
    ("a spider", "an arachnid", "arachnids"),
    ("a scorpion", "an arachnid", "arachnids"),
    ("a crab", "a crustacean", "crustaceans"),
    ("a lobster", "a crustacean", "crustaceans"),
    ("a mushroom", "a fungus", "fungi"),
    ("yeast", "a fungus", "fungi"),
    ("a rose", "a flowering plant", "flowering plants"),
    ("gold", "a metal", "metals"),
    ("iron", "a metal", "metals"),
    ("copper", "a metal", "metals"),
    ("silver", "a metal", "metals"),
    ("diamond", "a mineral", "minerals"),
    ("quartz", "a mineral", "minerals"),
    ("water", "a compound", "chemical compounds"),
    ("salt", "a compound", "chemical compounds"),
    ("oxygen", "an element", "chemical elements"),
    ("hydrogen", "an element", "chemical elements"),
    ("carbon", "an element", "chemical elements"),
    ("the sun", "a star", "stars"),
    ("Sirius", "a star", "stars"),
    ("Earth", "a planet", "planets"),
    ("Mars", "a planet", "planets"),
    ("the moon", "a natural satellite", "natural satellites"),
    ("a virus", "a microorganism", "microorganisms"),
    ("a bacterium", "a microorganism", "microorganisms"),
    ("wheat", "a cereal grain", "cereal grains"),
    ("rice", "a cereal grain", "cereal grains"),
    ("a piano", "a musical instrument", "musical instruments"),
    ("a violin", "a stringed instrument", "stringed instruments"),
    ("a trumpet", "a brass instrument", "brass instruments"),
    ("a flute", "a wind instrument", "wind instruments"),
    ("a triangle", "a polygon", "polygons"),
    ("a square", "a quadrilateral", "quadrilaterals"),
    ("a cube", "a polyhedron", "polyhedra"),
    ("chess", "a board game", "board games"),
    ("poker", "a card game", "card games"),
    ("a novel", "a work of fiction", "works of fiction"),
    ("a sonnet", "a type of poem", "types of poems"),
    ("French", "a Romance language", "Romance languages"),
    ("Spanish", "a Romance language", "Romance languages"),
    ("Italian", "a Romance language", "Romance languages"),
    ("English", "a Germanic language", "Germanic languages"),
    ("German", "a Germanic language", "Germanic languages"),
    ("Dutch", "a Germanic language", "Germanic languages"),
    ("algebra", "a branch of mathematics", "branches of mathematics"),
    ("geometry", "a branch of mathematics", "branches of mathematics"),
    ("calculus", "a branch of mathematics", "branches of mathematics"),
    ("physics", "a natural science", "natural sciences"),
    ("chemistry", "a natural science", "natural sciences"),
    ("biology", "a natural science", "natural sciences"),
    ("a hammer", "a tool", "tools"),
    ("a screwdriver", "a tool", "tools"),
    ("rubber", "an insulator", "electrical insulators"),
    ("ice", "a solid", "solids"),
    ("steam", "a gas", "gases"),
    ("an apple", "a fruit", "fruits"),
    ("a banana", "a fruit", "fruits"),
    ("a carrot", "a vegetable", "vegetables"),
    ("a tomato", "a fruit", "fruits"),
    ("granite", "a rock", "rocks"),
    ("limestone", "a rock", "rocks"),
    ("sandstone", "a sedimentary rock", "sedimentary rocks"),
    ("marble", "a metamorphic rock", "metamorphic rocks"),
    ("basalt", "a volcanic rock", "volcanic rocks"),
    ("a hurricane", "a tropical storm", "tropical storms"),
    ("a tornado", "a weather phenomenon", "weather phenomena"),
    ("an earthquake", "a geological event", "geological events"),
    ("a tsunami", "a natural disaster", "natural disasters"),
    ("a sonata", "a musical composition", "musical compositions"),
    ("a symphony", "a musical composition", "musical compositions"),
    ("a haiku", "a type of poem", "types of poems"),
    ("a limerick", "a type of poem", "types of poems"),
    ("a democracy", "a form of government", "forms of government"),
    ("a monarchy", "a form of government", "forms of government"),
    ("a republic", "a form of government", "forms of government"),
    ("football", "a team sport", "team sports"),
    ("basketball", "a team sport", "team sports"),
    ("tennis", "a racket sport", "racket sports"),
    ("badminton", "a racket sport", "racket sports"),
    ("swimming", "an aquatic sport", "aquatic sports"),
    ("rugby", "a contact sport", "contact sports"),
    ("a documentary", "a film genre", "film genres"),
    ("a comedy", "a film genre", "film genres"),
    ("a tragedy", "a theatrical genre", "theatrical genres"),
]

for thing, indef_category, noun_category in CLASSIFICATIONS[:100]:
    groups.append({
        "category": "classification",
        "answer_type": "yes_no",
        "questions": [
            f"Is {thing} {indef_category}?",
            f"Is {thing} a type of {noun_category.rstrip('s')}?",
            f"Would you classify {thing} as {indef_category}?",
        ],
        "expected": "yes",
    })


# ---------------------------------------------------------------------------
# 7. Unit equivalences  (30 groups × 3 questions)
# ---------------------------------------------------------------------------

UNITS = [
    ("1 kilometer", "1000 meters"),
    ("1 kilogram", "1000 grams"),
    ("1 liter", "1000 milliliters"),
    ("1 hour", "60 minutes"),
    ("1 minute", "60 seconds"),
    ("1 day", "24 hours"),
    ("1 week", "7 days"),
    ("1 decade", "10 years"),
    ("1 century", "100 years"),
    ("1 meter", "100 centimeters"),
    ("1 foot", "12 inches"),
    ("1 pound", "16 ounces"),
    ("1 mile", "5280 feet"),
    ("1 dozen", "12 items"),
    ("1 gross", "144 items"),
    ("1 kilowatt", "1000 watts"),
    ("1 megabyte", "1024 kilobytes"),
    ("1 gigabyte", "1024 megabytes"),
    ("1 kilometer", "0.621 miles"),
    ("1 inch", "2.54 centimeters"),
    ("1 pound", "0.454 kilograms"),
    ("1 gallon", "4 quarts"),
    ("1 quart", "2 pints"),
    ("1 pint", "2 cups"),
    ("1 cup", "8 fluid ounces"),
    ("1 year", "12 months"),
    ("1 millennium", "1000 years"),
    ("1 metric ton", "1000 kilograms"),
    ("1 hectare", "10000 square meters"),
    ("1 acre", "43560 square feet"),
]

for unit_a, unit_b in UNITS:
    groups.append({
        "category": "unit_equivalence",
        "answer_type": "yes_no",
        "questions": [
            f"Is {unit_a} equal to {unit_b}?",
            f"Does {unit_a} equal {unit_b}?",
            f"Is {unit_b} the same as {unit_a}?",
        ],
        "expected": "yes",
    })


# ---------------------------------------------------------------------------
# 8. Chemical formulas  (20 groups × 2 questions)
# ---------------------------------------------------------------------------

CHEMICALS = [
    ("water", "H2O"),
    ("table salt", "NaCl"),
    ("carbon dioxide", "CO2"),
    ("oxygen", "O2"),
    ("hydrogen", "H2"),
    ("ammonia", "NH3"),
    ("methane", "CH4"),
    ("sulfuric acid", "H2SO4"),
    ("hydrochloric acid", "HCl"),
    ("sodium hydroxide", "NaOH"),
    ("calcium carbonate", "CaCO3"),
    ("nitrogen gas", "N2"),
    ("glucose", "C6H12O6"),
    ("ethanol", "C2H5OH"),
    ("ozone", "O3"),
    ("hydrogen peroxide", "H2O2"),
    ("nitric acid", "HNO3"),
    ("potassium chloride", "KCl"),
    ("calcium oxide", "CaO"),
    ("iron oxide", "Fe2O3"),
]

for compound, formula in CHEMICALS:
    groups.append({
        "category": "chemical_formula",
        "answer_type": "yes_no",
        "questions": [
            f"Is the chemical formula for {compound} {formula}?",
            f"Is {formula} the chemical formula for {compound}?",
        ],
        "expected": "yes",
    })


# ---------------------------------------------------------------------------
# 9. Capital retrieval — one-word answer  (100 groups × 3 questions)
# ---------------------------------------------------------------------------

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "capital_retrieval",
        "answer_type": "word",
        "questions": [
            f"What is the capital of {country}?",
            f"Which city is the capital of {country}?",
            f"Name the capital city of {country}.",
        ],
        "expected": capital,
    })


# ---------------------------------------------------------------------------
# 10. Arithmetic result — one-word answer  (100 groups × 3 questions)
# ---------------------------------------------------------------------------

seen_result_sums: set[tuple] = set()
while len(seen_result_sums) < 100:
    a = rng.randint(2, 30)
    b = rng.randint(2, 30)
    key = (min(a, b), max(a, b))
    if key in seen_result_sums:
        continue
    seen_result_sums.add(key)
    c = a + b
    groups.append({
        "category": "arithmetic_result",
        "answer_type": "word",
        "questions": [
            f"What is {a} plus {b}?",
            f"What does {a} plus {b} equal?",
            f"What is the sum of {a} and {b}?",
        ],
        "expected": str(c),
    })


# ---------------------------------------------------------------------------
# Summary and output
# ---------------------------------------------------------------------------

# Assign IDs
for group_id, group in enumerate(groups):
    group["id"] = group_id

# Count by category
counts: dict[str, int] = {}
for group in groups:
    counts[group["category"]] = counts.get(group["category"], 0) + 1

total_questions = sum(len(g["questions"]) for g in groups)

yes_count = sum(1 for g in groups if g.get("expected") == "yes")
no_count  = sum(1 for g in groups if g.get("expected") == "no")
word_count = sum(1 for g in groups if g.get("answer_type") == "word")

print(f"Generated {len(groups)} equivalence groups ({total_questions} total questions)")
print(f"  yes: {yes_count}  no: {no_count}  word: {word_count}")
print()
print("By category:")
for category, count in sorted(counts.items()):
    print(f"  {category:<30} {count:>4} groups")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as output_file:
    json.dump(groups, output_file, indent=2, ensure_ascii=False)

print(f"\nSaved to {OUTPUT_PATH}")
