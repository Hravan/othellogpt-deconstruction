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
    # original 50
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
    # additional 50 to reach 100 total
    ("Archimedes", "discovered", "the principle of buoyancy"),
    ("Galileo Galilei", "invented", "the thermoscope"),
    ("Nicolaus Copernicus", "proposed", "the heliocentric model of the solar system"),
    ("Johannes Kepler", "formulated", "the laws of planetary motion"),
    ("Christiaan Huygens", "invented", "the wave theory of light"),
    ("Robert Hooke", "discovered", "Hooke's law of elasticity"),
    ("Joseph Priestley", "discovered", "oxygen"),
    ("John Dalton", "proposed", "the atomic theory of matter"),
    ("Amedeo Avogadro", "proposed", "Avogadro's law of gases"),
    ("William Harvey", "discovered", "the circulation of blood"),
    ("Carl Linnaeus", "developed", "the binomial nomenclature system"),
    ("Gregor Mendel", "discovered", "the laws of heredity"),
    ("Louis Pasteur", "invented", "the rabies vaccine"),
    ("Robert Koch", "discovered", "the tuberculosis bacillus"),
    ("Wilhelm Röntgen", "discovered", "X-rays"),
    ("Henri Becquerel", "discovered", "radioactivity"),
    ("J.J. Thomson", "discovered", "the electron"),
    ("Albert Einstein", "proposed", "the theory of special relativity"),
    ("Erwin Schrödinger", "formulated", "the wave equation of quantum mechanics"),
    ("Paul Dirac", "formulated", "the Dirac equation"),
    ("Francis Crick and James Watson", "discovered", "the double helix structure of DNA"),
    ("Rosalind Franklin", "produced", "the X-ray diffraction image of DNA"),
    ("Frederick Sanger", "developed", "the method for sequencing proteins"),
    ("Karl Landsteiner", "discovered", "the ABO blood group system"),
    ("Ilya Mechnikov", "discovered", "phagocytosis"),
    ("Santiago Ramón y Cajal", "described", "the structure of neurons"),
    ("Ivan Pavlov", "discovered", "the conditioned reflex"),
    ("Sigmund Freud", "developed", "psychoanalysis"),
    ("Charles Babbage", "designed", "the difference engine"),
    ("George Boole", "developed", "Boolean algebra"),
    ("John von Neumann", "proposed", "the stored-program computer architecture"),
    ("Claude Shannon", "formulated", "information theory"),
    ("Grace Hopper", "developed", "the first compiler"),
    ("Bjarne Stroustrup", "created", "the C++ programming language"),
    ("Guido van Rossum", "created", "the Python programming language"),
    ("James Gosling", "created", "the Java programming language"),
    ("Larry Page and Sergey Brin", "invented", "the PageRank algorithm"),
    ("Léon Foucault", "invented", "the Foucault pendulum"),
    ("James Joule", "discovered", "the mechanical equivalent of heat"),
    ("William Thomson", "formulated", "the laws of thermodynamics"),
    ("Rudolf Clausius", "introduced", "the concept of entropy"),
    ("James Chadwick", "discovered", "the neutron"),
    ("Lise Meitner", "explained", "the process of nuclear fission"),
    ("Glenn Seaborg", "discovered", "plutonium"),
    ("Hedy Lamarr", "co-invented", "frequency-hopping spread spectrum"),
    ("Nicolas Appert", "invented", "the canning process for food preservation"),
    ("Charles Goodyear", "invented", "the vulcanization of rubber"),
    ("George Westinghouse", "invented", "the railway air brake"),
    ("Philo Farnsworth", "invented", "the electronic television"),
    ("John Bardeen", "invented", "the transistor"),
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
    # 3 more to reach exactly 100
    ("a thesis", "an academic work", "academic works"),
    ("a sonata", "a musical form", "musical forms"),
    ("platinum", "a metal", "metals"),
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
    # original 30
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
    # additional 70 to reach 100 total
    ("1 terabyte", "1024 gigabytes"),
    ("1 milligram", "0.001 grams"),
    ("1 microgram", "0.001 milligrams"),
    ("1 millimeter", "0.1 centimeters"),
    ("1 centimeter", "10 millimeters"),
    ("1 decimeter", "10 centimeters"),
    ("1 nanometer", "0.001 micrometers"),
    ("1 micrometer", "1000 nanometers"),
    ("1 light-year", "9.461 trillion kilometers"),
    ("1 astronomical unit", "149.6 million kilometers"),
    ("1 parsec", "3.086 trillion kilometers"),
    ("1 nautical mile", "1.852 kilometers"),
    ("1 league", "3 miles"),
    ("1 fathom", "6 feet"),
    ("1 yard", "3 feet"),
    ("1 furlong", "220 yards"),
    ("1 chain", "22 yards"),
    ("1 rod", "5.5 yards"),
    ("1 stone", "14 pounds"),
    ("1 short ton", "2000 pounds"),
    ("1 long ton", "2240 pounds"),
    ("1 ounce", "28.35 grams"),
    ("1 troy ounce", "31.10 grams"),
    ("1 carat", "0.2 grams"),
    ("1 grain", "0.0648 grams"),
    ("1 teaspoon", "5 milliliters"),
    ("1 tablespoon", "3 teaspoons"),
    ("1 fluid ounce", "29.57 milliliters"),
    ("1 barrel", "42 gallons"),
    ("1 bushel", "8 gallons"),
    ("1 peck", "2 gallons"),
    ("1 second", "1000 milliseconds"),
    ("1 millisecond", "1000 microseconds"),
    ("1 microsecond", "1000 nanoseconds"),
    ("1 fortnight", "2 weeks"),
    ("1 score", "20 years"),
    ("1 megawatt", "1000 kilowatts"),
    ("1 gigawatt", "1000 megawatts"),
    ("1 milliwatt", "0.001 watts"),
    ("1 joule", "1 watt-second"),
    ("1 kilojoule", "1000 joules"),
    ("1 megajoule", "1000 kilojoules"),
    ("1 calorie", "4.184 joules"),
    ("1 kilocalorie", "1000 calories"),
    ("1 kilowatt-hour", "3.6 megajoules"),
    ("1 electron volt", "1.602 times 10 to the negative 19 joules"),
    ("1 pascal", "1 newton per square meter"),
    ("1 kilopascal", "1000 pascals"),
    ("1 megapascal", "1000 kilopascals"),
    ("1 bar", "100000 pascals"),
    ("1 atmosphere", "101325 pascals"),
    ("1 millimeter of mercury", "133.32 pascals"),
    ("1 newton", "1 kilogram meter per second squared"),
    ("1 kilonewton", "1000 newtons"),
    ("1 ampere", "1 coulomb per second"),
    ("1 volt", "1 joule per coulomb"),
    ("1 ohm", "1 volt per ampere"),
    ("1 farad", "1 coulomb per volt"),
    ("1 henry", "1 volt-second per ampere"),
    ("1 tesla", "1 weber per square meter"),
    ("1 weber", "1 volt-second"),
    ("1 hertz", "1 cycle per second"),
    ("1 kilohertz", "1000 hertz"),
    ("1 megahertz", "1000 kilohertz"),
    ("1 gigahertz", "1000 megahertz"),
    ("1 square kilometer", "100 hectares"),
    ("1 square mile", "640 acres"),
    ("1 square foot", "144 square inches"),
    ("1 cubic meter", "1000 liters"),
    ("1 cubic foot", "28.317 liters"),
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
    # original 20
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
    # additional 80 to reach 100 total
    ("acetic acid", "CH3COOH"),
    ("acetone", "C3H6O"),
    ("benzene", "C6H6"),
    ("sodium carbonate", "Na2CO3"),
    ("sodium bicarbonate", "NaHCO3"),
    ("potassium permanganate", "KMnO4"),
    ("calcium chloride", "CaCl2"),
    ("magnesium oxide", "MgO"),
    ("aluminum oxide", "Al2O3"),
    ("silicon dioxide", "SiO2"),
    ("phosphoric acid", "H3PO4"),
    ("sulfur dioxide", "SO2"),
    ("sulfur trioxide", "SO3"),
    ("nitrogen dioxide", "NO2"),
    ("nitrous oxide", "N2O"),
    ("carbon monoxide", "CO"),
    ("chlorine gas", "Cl2"),
    ("fluorine gas", "F2"),
    ("bromine", "Br2"),
    ("iodine", "I2"),
    ("sodium chlorate", "NaClO3"),
    ("potassium nitrate", "KNO3"),
    ("calcium sulfate", "CaSO4"),
    ("magnesium sulfate", "MgSO4"),
    ("copper sulfate", "CuSO4"),
    ("zinc oxide", "ZnO"),
    ("zinc sulfate", "ZnSO4"),
    ("iron sulfate", "FeSO4"),
    ("iron chloride", "FeCl3"),
    ("silver nitrate", "AgNO3"),
    ("barium sulfate", "BaSO4"),
    ("lead oxide", "PbO"),
    ("mercury chloride", "HgCl2"),
    ("ammonium chloride", "NH4Cl"),
    ("ammonium nitrate", "NH4NO3"),
    ("ammonium sulfate", "(NH4)2SO4"),
    ("sodium sulfate", "Na2SO4"),
    ("sodium nitrate", "NaNO3"),
    ("potassium hydroxide", "KOH"),
    ("magnesium hydroxide", "Mg(OH)2"),
    ("calcium hydroxide", "Ca(OH)2"),
    ("aluminum hydroxide", "Al(OH)3"),
    ("formic acid", "HCOOH"),
    ("oxalic acid", "C2H2O4"),
    ("citric acid", "C6H8O7"),
    ("lactic acid", "C3H6O3"),
    ("sucrose", "C12H22O11"),
    ("fructose", "C6H12O6"),
    ("starch", "C6H10O5"),
    ("urea", "CO(NH2)2"),
    ("formaldehyde", "CH2O"),
    ("acetaldehyde", "C2H4O"),
    ("propane", "C3H8"),
    ("butane", "C4H10"),
    ("ethylene", "C2H4"),
    ("acetylene", "C2H2"),
    ("toluene", "C7H8"),
    ("naphthalene", "C10H8"),
    ("chloroform", "CHCl3"),
    ("carbon tetrachloride", "CCl4"),
    ("diethyl ether", "C4H10O"),
    ("ethylene glycol", "C2H6O2"),
    ("glycerol", "C3H8O3"),
    ("aspirin", "C9H8O4"),
    ("caffeine", "C8H10N4O2"),
    ("cholesterol", "C27H46O"),
    ("adenine", "C5H5N5"),
    ("guanine", "C5H5N5O"),
    ("thymine", "C5H6N2O2"),
    ("cytosine", "C4H5N3O"),
    ("acrylic acid", "C3H4O2"),
    ("acetamide", "C2H5NO"),
    ("aniline", "C6H7N"),
    ("phenol", "C6H6O"),
    ("ethane", "C2H6"),
    ("pentane", "C5H12"),
    ("hexane", "C6H14"),
    ("cyclohexane", "C6H12"),
    ("dinitrogen tetroxide", "N2O4"),
    ("phosphorus pentachloride", "PCl5"),
]

for compound, formula in CHEMICALS:
    groups.append({
        "category": "chemical_formula",
        "answer_type": "yes_no",
        "questions": [
            f"Is the chemical formula for {compound} {formula}?",
            f"Is {formula} the chemical formula for {compound}?",
            f"Does {compound} have the chemical formula {formula}?",
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
# 11. Large-number arithmetic commutativity  (200 yes + 100 no groups × 3 questions)
#
# Same structure as arithmetic_order but with 3-digit numbers (100–999).
# Models that memorised small-number arithmetic facts may still fail here.
# ---------------------------------------------------------------------------

seen_large_sums: set[tuple] = set()
while len(seen_large_sums) < 200:
    a = rng.randint(100, 999)
    b = rng.randint(100, 999)
    key = (min(a, b), max(a, b))
    if key in seen_large_sums:
        continue
    seen_large_sums.add(key)
    c = a + b
    groups.append({
        "category": "arithmetic_large",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} plus {b} equal to {c}?",
            f"Does {b} plus {a} equal {c}?",
            f"Is {c} the sum of {a} and {b}?",
        ],
        "expected": "yes",
    })

seen_large_no_sums: set[tuple] = set()
while len(seen_large_no_sums) < 100:
    a = rng.randint(100, 999)
    b = rng.randint(100, 999)
    key = (min(a, b), max(a, b))
    if key in seen_large_no_sums or key in seen_large_sums:
        continue
    seen_large_no_sums.add(key)
    c = a + b
    wrong_c = c + rng.choice([-3, -2, -1, 1, 2, 3])
    groups.append({
        "category": "arithmetic_large",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} plus {b} equal to {wrong_c}?",
            f"Does {b} plus {a} equal {wrong_c}?",
            f"Is {wrong_c} the sum of {a} and {b}?",
        ],
        "expected": "no",
    })


# ---------------------------------------------------------------------------
# 12. Multiplication commutativity  (200 yes + 100 no groups × 3 questions)
#
# "Is A times B equal to C?" ↔ "Is B times A equal to C?" ↔
# "Is C the product of A and B?"
# ---------------------------------------------------------------------------

seen_products: set[tuple] = set()
while len(seen_products) < 200:
    a = rng.randint(3, 50)
    b = rng.randint(3, 50)
    key = (min(a, b), max(a, b))
    if key in seen_products:
        continue
    seen_products.add(key)
    c = a * b
    groups.append({
        "category": "multiplication_order",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} times {b} equal to {c}?",
            f"Does {b} times {a} equal {c}?",
            f"Is {c} the product of {a} and {b}?",
        ],
        "expected": "yes",
    })

seen_no_products: set[tuple] = set()
while len(seen_no_products) < 100:
    a = rng.randint(3, 50)
    b = rng.randint(3, 50)
    key = (min(a, b), max(a, b))
    if key in seen_no_products or key in seen_products:
        continue
    seen_no_products.add(key)
    c = a * b
    wrong_c = c + rng.choice([-2, -1, 1, 2]) * rng.randint(1, 3)
    groups.append({
        "category": "multiplication_order",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} times {b} equal to {wrong_c}?",
            f"Does {b} times {a} equal {wrong_c}?",
            f"Is {wrong_c} the product of {a} and {b}?",
        ],
        "expected": "no",
    })


# ---------------------------------------------------------------------------
# 13. Subtraction-addition equivalence  (150 yes + 75 no groups × 3 questions)
#
# "Is A minus B equal to C?" ↔ "Is C plus B equal to A?" ↔
# "Is C the result of subtracting B from A?"
# Tests whether models recognise that subtraction and addition are inverses.
# ---------------------------------------------------------------------------

seen_subtractions: set[tuple] = set()
while len(seen_subtractions) < 150:
    a = rng.randint(20, 200)
    b = rng.randint(5, a - 5)
    key = (a, b)
    if key in seen_subtractions:
        continue
    seen_subtractions.add(key)
    c = a - b
    groups.append({
        "category": "subtraction_equivalence",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} minus {b} equal to {c}?",
            f"Is {c} plus {b} equal to {a}?",
            f"Is {c} the result of subtracting {b} from {a}?",
        ],
        "expected": "yes",
    })

seen_no_subtractions: set[tuple] = set()
while len(seen_no_subtractions) < 75:
    a = rng.randint(20, 200)
    b = rng.randint(5, a - 5)
    key = (a, b)
    if key in seen_no_subtractions or key in seen_subtractions:
        continue
    seen_no_subtractions.add(key)
    c = a - b
    wrong_c = c + rng.choice([-2, -1, 1, 2])
    groups.append({
        "category": "subtraction_equivalence",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} minus {b} equal to {wrong_c}?",
            f"Is {wrong_c} plus {b} equal to {a}?",
            f"Is {wrong_c} the result of subtracting {b} from {a}?",
        ],
        "expected": "no",
    })


# ---------------------------------------------------------------------------
# 14. Convoluted arithmetic phrasings  (150 yes + 75 no groups × 5 questions)
#
# Five increasingly indirect phrasings of the same addition fact.
# Tests whether models maintain consistency as surface form grows more distant
# from a direct equation.
# ---------------------------------------------------------------------------

seen_convoluted: set[tuple] = set()
while len(seen_convoluted) < 150:
    a = rng.randint(5, 80)
    b = rng.randint(5, 80)
    key = (min(a, b), max(a, b))
    if key in seen_convoluted:
        continue
    seen_convoluted.add(key)
    c = a + b
    groups.append({
        "category": "arithmetic_convoluted",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} plus {b} equal to {c}?",
            f"When you add {a} to {b}, do you get {c}?",
            f"If you start with {a} and add {b}, is the result {c}?",
            f"Does adding {b} to {a} yield {c}?",
            f"Is {c} what you get when you combine {a} and {b}?",
        ],
        "expected": "yes",
    })

seen_no_convoluted: set[tuple] = set()
while len(seen_no_convoluted) < 75:
    a = rng.randint(5, 80)
    b = rng.randint(5, 80)
    key = (min(a, b), max(a, b))
    if key in seen_no_convoluted or key in seen_convoluted:
        continue
    seen_no_convoluted.add(key)
    c = a + b
    wrong_c = c + rng.choice([-2, -1, 1, 2])
    groups.append({
        "category": "arithmetic_convoluted",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} plus {b} equal to {wrong_c}?",
            f"When you add {a} to {b}, do you get {wrong_c}?",
            f"If you start with {a} and add {b}, is the result {wrong_c}?",
            f"Does adding {b} to {a} yield {wrong_c}?",
            f"Is {wrong_c} what you get when you combine {a} and {b}?",
        ],
        "expected": "no",
    })


# ---------------------------------------------------------------------------
# 15. Double negation  (100 groups × 3 questions)
#
# "Is X the capital of Y?" ↔ "Is it false that X is not the capital of Y?" ↔
# "Is it not the case that X is not the capital of Y?"
# Tests whether models maintain semantic consistency through double negation.
# ---------------------------------------------------------------------------

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "double_negation",
        "answer_type": "yes_no",
        "questions": [
            f"Is {capital} the capital of {country}?",
            f"Is it false that {capital} is not the capital of {country}?",
            f"Is it not the case that {capital} is not the capital of {country}?",
        ],
        "expected": "yes",
    })

# Wrong capitals — expected "no" (all three questions should agree on "no")
for country, _correct_capital, wrong_capital in wrong_capital_pairs[:100]:
    groups.append({
        "category": "double_negation",
        "answer_type": "yes_no",
        "questions": [
            f"Is {wrong_capital} the capital of {country}?",
            f"Is it false that {wrong_capital} is not the capital of {country}?",
            f"Is it not the case that {wrong_capital} is not the capital of {country}?",
        ],
        "expected": "no",
    })


# ---------------------------------------------------------------------------
# 16. Comparison convoluted  (150 yes + 75 no groups × 4 questions)
#
# "Is A greater than B?" ↔ "Is B smaller than A?" ↔ "Does A exceed B?" ↔
# "Is it true that A is not less than or equal to B?"
# Extends comparison_symmetric with more convoluted phrasings.
# ---------------------------------------------------------------------------

seen_convoluted_comparisons: set[tuple] = set()
while len(seen_convoluted_comparisons) < 150:
    a = rng.randint(2, 500)
    b = rng.randint(2, 500)
    if a == b:
        continue
    large, small = max(a, b), min(a, b)
    key = (large, small)
    if key in seen_convoluted_comparisons:
        continue
    seen_convoluted_comparisons.add(key)
    groups.append({
        "category": "comparison_convoluted",
        "answer_type": "yes_no",
        "questions": [
            f"Is {large} greater than {small}?",
            f"Is {small} smaller than {large}?",
            f"Does {large} exceed {small}?",
            f"Is it true that {large} is not less than or equal to {small}?",
        ],
        "expected": "yes",
    })

seen_no_convoluted_comparisons: set[tuple] = set()
while len(seen_no_convoluted_comparisons) < 75:
    a = rng.randint(2, 500)
    b = rng.randint(2, 500)
    if a == b:
        continue
    large, small = max(a, b), min(a, b)
    key = (large, small)
    if key in seen_no_convoluted_comparisons or key in seen_convoluted_comparisons:
        continue
    seen_no_convoluted_comparisons.add(key)
    groups.append({
        "category": "comparison_convoluted",
        "answer_type": "yes_no",
        "questions": [
            f"Is {small} greater than {large}?",
            f"Is {large} smaller than {small}?",
            f"Does {small} exceed {large}?",
            f"Is it true that {small} is not less than or equal to {large}?",
        ],
        "expected": "no",
    })


# ---------------------------------------------------------------------------
# 17. Negation depth experiment  (100 groups × 4 questions, depths 0/2/3/4)
#
# Same capital facts as double_negation but extended to depth 3 and 4.
# Each group has one question per negation depth (0, 2, 3, 4):
#   depth 0: P                = "Is X the capital of Y?"                                           → yes
#   depth 2: ¬¬P = P         = "Is it false that X is NOT the capital of Y?"                      → yes
#   depth 3: ¬³P = ¬P        = "Is it not the case that it is false that X is NOT the capital of Y?" → no
#   depth 4: ¬⁴P = P         = "Is it false that it is not the case that it is false that X is NOT the capital of Y?" → yes
#
# NOTE: CR is not meaningful here — a perfect model would have CR>0 because
# depth-3 has a different answer than the others. Use per-depth accuracy
# (see negation_finetune.py report_per_depth_accuracy) instead.
# ---------------------------------------------------------------------------

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "negation_depth",
        "answer_type": "yes_no",
        "questions": [
            f"Is {capital} the capital of {country}?",
            f"Is it false that {capital} is not the capital of {country}?",
            f"Is it not the case that it is false that {capital} is not the capital of {country}?",
            f"Is it false that it is not the case that it is false that {capital} is not the capital of {country}?",
        ],
        "expected": "yes",
    })


# ---------------------------------------------------------------------------
# 18. Nested arithmetic negation  (100 yes + 50 no groups × 4 questions)
#
# Combines arithmetic verification with negation depth.
#   depth 1: "Is A+B=C?"
#   depth 2: "Is it false that A+B is not equal to C?"
#   depth 3: "Is it not the case that it is false that A+B equals C?"
#   depth 4: "Is it false that it is not the case that it is false that A+B is not equal to C?"
# ---------------------------------------------------------------------------

seen_neg_arith: set[tuple] = set()
while len(seen_neg_arith) < 100:
    a = rng.randint(2, 60)
    b = rng.randint(2, 60)
    key = (min(a, b), max(a, b))
    if key in seen_neg_arith:
        continue
    seen_neg_arith.add(key)
    c = a + b
    groups.append({
        "category": "negation_arithmetic",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} plus {b} equal to {c}?",
            f"Is it false that {a} plus {b} is not equal to {c}?",
            f"Is it not the case that it is false that {a} plus {b} equals {c}?",
            f"Is it false that it is not the case that it is false that {a} plus {b} is not equal to {c}?",
        ],
        "expected": "yes",
    })

seen_neg_arith_no: set[tuple] = set()
while len(seen_neg_arith_no) < 50:
    a = rng.randint(2, 60)
    b = rng.randint(2, 60)
    key = (min(a, b), max(a, b))
    if key in seen_neg_arith_no or key in seen_neg_arith:
        continue
    seen_neg_arith_no.add(key)
    c = a + b
    wrong_c = c + rng.choice([-2, -1, 1, 2])
    groups.append({
        "category": "negation_arithmetic",
        "answer_type": "yes_no",
        "questions": [
            f"Is {a} plus {b} equal to {wrong_c}?",
            f"Is it false that {a} plus {b} is not equal to {wrong_c}?",
            f"Is it not the case that it is false that {a} plus {b} equals {wrong_c}?",
            f"Is it false that it is not the case that it is false that {a} plus {b} is not equal to {wrong_c}?",
        ],
        "expected": "no",
    })


# ---------------------------------------------------------------------------
# 19. Contrastive negation  (100 groups × 3 questions)
#
# Tests whether the model understands negation in contrastive contexts:
#   "Is X the capital of Y, not Z?"  (affirmative with explicit contrast)
#   "Is it X, not Z, that is the capital of Y?"
#   "Is the capital of Y X rather than Z?"
# All three mean the same thing given that X is the capital and Z is not.
# ---------------------------------------------------------------------------

shuffled_for_contrast = CAPITALS[:]
rng.shuffle(shuffled_for_contrast)
contrast_pairs: list[tuple[str, str, str]] = []
for country, capital in CAPITALS[:100]:
    for _, other_capital in shuffled_for_contrast:
        if other_capital != capital:
            contrast_pairs.append((country, capital, other_capital))
            break

for country, capital, other_capital in contrast_pairs[:100]:
    groups.append({
        "category": "contrastive_negation",
        "answer_type": "yes_no",
        "questions": [
            f"Is {capital}, not {other_capital}, the capital of {country}?",
            f"Is it {capital}, not {other_capital}, that is the capital of {country}?",
            f"Is the capital of {country} {capital} rather than {other_capital}?",
        ],
        "expected": "yes",
    })


# ---------------------------------------------------------------------------
# 20. Negation depth by fixed count  (100 groups × 3 questions each, depths 0–6)
#
# Each category contains 3 distinct phrasings at exactly N negation operators,
# all logically equivalent (same answer). This allows SS/CR to be computed
# within each negation depth, and provides training/test data for fine-tuning.
#
# Answer pattern for correct capitals (P = "X is the capital of Y"):
#   depth 0: P      → yes   depth 1: ¬P     → no
#   depth 2: ¬²P=P  → yes   depth 3: ¬³P=¬P → no
#   depth 4: ¬⁴P=P  → yes   depth 5: ¬⁵P=¬P → no
#   depth 6: ¬⁶P=P  → yes
#
# Negation operators used: "is not the case that", "is false that",
# "is not true that", and embedded "is not the capital of".
# ---------------------------------------------------------------------------

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "negation_depth_0",
        "answer_type": "yes_no",
        "questions": [
            f"Is {capital} the capital of {country}?",
            f"Does {capital} serve as the capital of {country}?",
            f"Is {country}'s capital {capital}?",
        ],
        "expected": "yes",
    })

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "negation_depth_1",
        "answer_type": "yes_no",
        "questions": [
            f"Is it not the case that {capital} is the capital of {country}?",
            f"Is it false that {capital} is the capital of {country}?",
            f"Is it not true that {capital} is the capital of {country}?",
        ],
        "expected": "no",
    })

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "negation_depth_2",
        "answer_type": "yes_no",
        "questions": [
            f"Is it not the case that {capital} is not the capital of {country}?",
            f"Is it false that {capital} is not the capital of {country}?",
            f"Is it not the case that it is false that {capital} is the capital of {country}?",
        ],
        "expected": "yes",
    })

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "negation_depth_3",
        "answer_type": "yes_no",
        "questions": [
            f"Is it not the case that it is false that {capital} is not the capital of {country}?",
            f"Is it false that it is not the case that {capital} is not the capital of {country}?",
            f"Is it not the case that it is not the case that {capital} is not the capital of {country}?",
        ],
        "expected": "no",
    })

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "negation_depth_4",
        "answer_type": "yes_no",
        "questions": [
            f"Is it false that it is not the case that it is false that {capital} is not the capital of {country}?",
            f"Is it not the case that it is false that it is not the case that {capital} is not the capital of {country}?",
            f"Is it not the case that it is not the case that it is not the case that {capital} is not the capital of {country}?",
        ],
        "expected": "yes",
    })

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "negation_depth_5",
        "answer_type": "yes_no",
        "questions": [
            f"Is it not the case that it is false that it is not the case that it is false that {capital} is not the capital of {country}?",
            f"Is it false that it is not the case that it is false that it is not the case that {capital} is not the capital of {country}?",
            f"Is it not the case that it is not the case that it is not the case that it is not the case that {capital} is not the capital of {country}?",
        ],
        "expected": "no",
    })

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "negation_depth_6",
        "answer_type": "yes_no",
        "questions": [
            f"Is it false that it is not the case that it is false that it is not the case that it is false that {capital} is not the capital of {country}?",
            f"Is it not the case that it is false that it is not the case that it is false that it is not the case that {capital} is not the capital of {country}?",
            f"Is it not the case that it is not the case that it is not the case that it is not the case that it is not the case that {capital} is not the capital of {country}?",
        ],
        "expected": "yes",
    })


# ---------------------------------------------------------------------------
# 21. Negation parity groups  (100 even + 100 odd groups × 4/3 questions)
#
# Tests cross-depth consistency within the same parity class.
# negation_even: one phrasing per even depth (0,2,4,6), all answer "yes".
# negation_odd:  one phrasing per odd depth  (1,3,5),   all answer "no".
# A model that correctly applies the negation rule should have SS≈0 and CR=0
# within each parity group — all phrasings are logically equivalent.
# ---------------------------------------------------------------------------

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "negation_even",
        "answer_type": "yes_no",
        "questions": [
            f"Is {capital} the capital of {country}?",
            f"Is it not the case that {capital} is not the capital of {country}?",
            f"Is it false that it is not the case that it is false that {capital} is not the capital of {country}?",
            f"Is it false that it is not the case that it is false that it is not the case that it is false that {capital} is not the capital of {country}?",
        ],
        "expected": "yes",
    })

for country, capital in CAPITALS[:100]:
    groups.append({
        "category": "negation_odd",
        "answer_type": "yes_no",
        "questions": [
            f"Is it not the case that {capital} is the capital of {country}?",
            f"Is it not the case that it is false that {capital} is not the capital of {country}?",
            f"Is it not the case that it is false that it is not the case that it is false that {capital} is not the capital of {country}?",
        ],
        "expected": "no",
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
