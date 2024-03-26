import re

from utils.category import convert_categories_name_to_index
from utils.category_type import CategoryType


class HeuristicGenerator:
    """
    Class for generating heuristic labels based on text content.
    """

    def __init__(self, category_type: CategoryType):
        """
        Initialize the HeuristicGenerator object.

        :param: category_type (CategoryType): Type of category.
        """
        self._category_type = category_type

    def get_heuristic_labels(self, text: str):
        """
        Get heuristic labels based on the input text.

        :param: text (str): Input text.
        :returns: list: List of heuristic labels.
        """
        categories = []

        # Add keyword heuristics
        categories.extend(self.get_keyword_heuristics(text))

        # Add years heuristics
        categories.extend(self.get_years_heuristics(text))

        category_indexes = []
        for category in categories:
            category_indexes.append(convert_categories_name_to_index(category, self._category_type))

        labels = list(set(convert_categories_name_to_index(categories, self._category_type)))
        return labels

    def get_keyword_heuristics(self, text: str):
        """
        Get keyword-based heuristic labels based on the input text.

        :param: text (str): Input text.
        :returns: list: List of heuristic labels.
        """
        if self._category_type == CategoryType.BIG:
            return get_keyword_heuristics_extended(text)
        return get_keyword_heuristics_shortened(text)

    def get_years_heuristics(self, text: str):
        """
        Get years-based heuristic labels based on the input text.

        :params: text (str): Input text.
        :returns: list: List of heuristic labels.
        """
        if self._category_type == CategoryType.BIG:
            return get_years_heuristics_extended(text)
        return get_years_heuristics_shortened(text)


def get_years_heuristics_shortened(text):
    """
    Get years-based heuristic labels for shortened category type.

    :param: text (str): Input text.
    :returns: list: List of heuristic labels.
    """
    year_count = count_years_in_text(text)
    categories = []

    if year_count > 5:
        categories.append("History")

    if year_count >= 2:
        categories.append("People")
        categories.append("Society")

    return categories


def get_years_heuristics_extended(text):
    """
    Get years-based heuristic labels for extended category type.

    :param: text (str): Input text.
    :returns: list: List of heuristic labels.
    """
    year_count = count_years_in_text(text)
    categories = []

    if year_count > 5:
        categories.append("History")

    if year_count >= 2:
        categories.append("Biography")
        categories.append("Education")
        categories.append("Politics")

    return categories


def get_keyword_heuristics_shortened(text):
    """
    Get keyword-based heuristic labels for shortened category type.

    :param: text (str): Input text.
    :returns: list: List of heuristic labels.
    """
    categories = []

    # Culture and Arts
    match = re.search(
        r'(\b(tone|press|art|set|line|lines|form|key|act)\b)'
        r'|(scholar|news|media|journalism|reporter|broadcast|television|radio|print|social media|internet'
        r'|advertising|communication|philosophy|philosopher|ethics|metaphysics|thinker|thought|logic|existence'
        r'|mind|truth|knowledge|reality|marality|reason|rationality|perception|painting|sculpture|gallery'
        r'|artist|exhibition|canvas|brush|palette|composition|perspective|color|texture|style|medium'
        r'|technique|expression|realism|surrealism|modern|theater|drama|playwright|performance|play|rehearsal'
        r'|costume|lighting|props|curtain|film|movie|director|cinema|actor|actress|scene|script|editing'
        r'|soundtrack|visual|dialogue|critic|premiere|award|sequel|screenplay|music|composer|orchestra|guitar'
        r'|song|singer|concert|melody|rhythm|harmony|instrument|vocals|sound|album|band|recording'
        r'|lyrics|beat|chord|chorus|soloist|improvisation|stage|audience|novel|poetry|literature|author|book'
        r'|writing|fable|fables|saga|poem|protagonist|antagonist|prose|character|plot|setting|theme|symbolism'
        r'|metaphor|genre|narrative|symbol|serie|episode|star|magazine)',
        text, re.IGNORECASE)

    if match:
        categories.append("Culture and Art")

    # Health
    match = re.search(
        r'\b(ice|sip|bar|win|run|lap|lose|wine|cup)\b'
        r'|(drink|beverage|coffee|cocktail|brewery|barista|alcohol|mix|glass|shake|stir|garnish|falvor'
        r'|liquor|champagn|soda|food|cuisine|cooking|restaurant|chef|recipe|dish|ingredients|flavors|taste|meal|appetizer'
        r'|dessert|spice|fresh|organic|medicine|medical|doctor|health|hospital|patient|treatment|disease'
        r'|symptom|diagonsis|injury|surgery|search|vaccine|therapy|clinic|specialist|laboratory|sport|soccer'
        r'|basketball|athlet|goal|captain|line-up|loan|league|winner|penalty|driver|tennis|football|baseball|player|competition|game|team|match|score|champion'
        r'|tournament|coach|train|success|victory|defeat|record|ball|overtime|field|court|referee|fans|stadium|season|club|draw|amateur)',
        text, re.IGNORECASE)

    if match:
        categories.append("Health")

    # History
    match = re.search(
        r'\b(war|era|date)\b'
        r'|(history|historical|ancient|timeline|event|period|government|empire|culture|tradition|heritage'
        r'|document|research|king|military|prince|century|troop|emperor|battle)',
        text, re.IGNORECASE)

    if match:
        categories.append("History")

    # Science
    match = re.search(
        r'\b(sea|air|sun|oak|rose|pet|ion|cat|map|base|egg|pine|bos|sus|eco|stem|lion|gene|fox|sand)\b|'
        r'(scholar|lake|aquatic|maritime|naval|beach|marine|environmental|sustainability|pollution'
        r'|sustainab|renewable|diodiversity|warming|greenhouse|deforestation|recycling|preservation|ocean'
        r'|footprint|impact|nature|landscape|environment|outdoors|wilderness|forest|river|wild|fauna'
        r'|biodiversity|flora|tree|flower|botany|gardening|horticulture|growth|leaves|roots|foliage'
        r'|nutrients|water|oxygen|carbon|pollination|reproduction|tulip|maple|cactus|orchid|animal'
        r'|wildlife|class|order|nest|vegetation|temperature|incub|predation|family|genus|zoology|creature|mammal'
        r'|reptile|amphibian|insect|predator|wood|garden'
        r'|prey|habitat|endangered|conservation|extinct|ecosystem|bio|adaption|camoflage|carnivore|herbivore'
        r'|omnivore|hibernation|noctural|diurnal|exotic|nesting|territory|zoo|fossil|dog|fish|bird|horse'
        r'|cow|pig|elephant|tiger|spider|griaffe|bear|rabbit|monkey|dolphin|shark|snake|penguin|koala|kanaroo'
        r'|wolf|owl|deer|camel|canis|felis|homo|humanities|philosophy|arts|literature|history|human|ethics'
        r'|language|civilization|cultur|exhibition|thinking|social science|sociology|anthropology|culture|psychology'
        r'|behavior|society|research|theory|data|analysis|socialogy|anthropologie|economics|political|gender'
        r'|globalization|power|social|geography|geographical|continent|geology|landform|topography|earth'
        r'|country|region|climate|terrain|resources|biome|latitude|longitude|population|urbanization|migration'
        r'|natural|plate|geo|cartography|mathematics|mathematical|algebra|geometry|calculus|equation'
        r'|mathematician|numbers|function|variable|theorem|proof|statistic|probability|matrix|derivative'
        r'|integral|polynom|graph|coordinate|prime|vector|chemical|compound|molecule|element|reaction|chemistry'
        r'|atom|chem|peroad|acid|solution|crystal|catalyst|electron|proton|neutron|isotope|redox'
        r'|oxid|reduction|polymer|eneregy|physics|physical|energy|quantum|mechanics|thermodynamics|particle'
        r'|matter|force|motion|velocity|mass|gravity|wave|light|magnetism|biology|biological|organism|ecology'
        r'|genetics|species|evolution|cell|dna|diversity|photosynthesis|food|chromosom|agriculture'
        r'|farming|crop|livestock|agronomy|harvest|farm|soil|irrigation|cultivation|plant|fertilizer|rural|star|moon|planet)',
        text, re.IGNORECASE)

    if match:
        categories.append("Science")

    # People
    match = re.search(
        r'\b(class|son)\b|'
        r'(education|educational|teaching|learning|school|curriculum|student|teach|know|skill|assessment'
        r'|instruction|university|biography|grade|life|memoir|autobiography|biographical|history|work|profile|person'
        r'|child|kid|adult|parent|friend|family|eductaion|career|archievements|legacy|influence|impact|timeline|events|challenges'
        r'|relationships|degree|worked|written|married|daughter|brother|sister|founder|father|mother)',
        text, re.IGNORECASE)

    if match:
        categories.append("People")

    # Religion
    match = re.search(
        r'\b(sin|hell)\b|'
        r'(religion|religious|faith|church|spirituality|belief|theology|god|prayer|spiritaulity|sacre|scripture'
        r'|mosque|temple|holy|ritual|salvation|heaven|priest|prophet|islam|hindu|christ|jude|jewish|buddhism'
        r'|bible|bishop)',
        text, re.IGNORECASE)

    if match:
        categories.append("Religion")

    # Society
    match = re.search(
        r'\b(law|rate)\b|'
        r'financ|banking|money|portfolio|stock|income|bank|expense|asset|liability|dividend|dollar'
        r'|euro|credit|corporate|company|entrepreneur|industry|enterprise|commerce|management|finance'
        r'|revenue|profit|invest|customer|gold|silver|billion|million|market|compet|strategy|employ|product|service|sustain|innovation'
        r'|risk|return|balance|liquidity|economics|economic|headquater|trade|economist|economy|business'
        r'|supply|demand|democrat|party|store'
        r'|gdp|inflation|unemployment|growth|tax|budget|investment|interest|finan|policy|regulation|captial'
        r'|consumer|production|politics|political|government|administration|leadership|democracy|legislation'
        r'|constitution|congress|president|senator|reprentative|cabinet|foreign|diplomacy|voter|lobbying'
        r'|debate|leisure|hobby|recreation|entertainment|pastime|relaxation|hobb|enjoyment|unwind|outdoor'
        r'|indoor|weekend|holiday|society|social|community|public|culture|norms|values|relationship|diversity'
        r'|equality|identity|class|gender|race|power|education|media|tradition|change|elect|organization',
        text, re.IGNORECASE)

    if match:
        categories.append("Society")

    # Technology
    match = re.search(
        r'\b(car|ship|port|jet|pi)\b|'
        r'(invention|inventor|gamma|parameter|plynomial|approximation|schema|graph|patent|discovery|creation|invent|prototype|advancement|breakthrough|development'
        r'|inventive|scirentific|creative|experiment|pioneering|revolution|transport|transportation|vehicle'
        r'|travel|railroad|aircraft|airship|airplane|logistic|shipping|road|freight|cargo|route|infrastructure|railway|delivery'
        r'|transit|traffic|engineering|engineer|mechanical|design|construction|optimization|efficiency|cost'
        r'|complexity|function|pressure|launch|performance|redundancy|analysis|quality|project|computer science|programming'
        r'|software|algorithm|comput|informatic|hardware|data|coding|system|network|artificial|machine|cyber'
        r'|operating|database|compiler|interface|experience|internet|cloud|program|robotic|graphic|info'
        r'|electronics|electronic|engine|gadget|device|technology|circuit|innovation|component|voltage|current'
        r'|resistance|capacitor|transistor|diode|micro|controller|conductor|amplifier|resistor|inductor'
        r'|digital|analyog|oscillator|phone|computer|laptop|mobile|andriod|touchscreen|apple|samsung|huawei)',
        text, re.IGNORECASE)

    if match:
        categories.append("Technology")

    # Geography and Places
    match = re.search(
        r'\b(river|km)\b|'
        r'cities|building|structure|architecture|edifice|construction|skyscraper|monument|design|foundation'
        r'|materials|facade|roof|wall|floor|mile|inland|window|interior|exterior|renovation|archtiect|engineer'
        r'|blueprint|resdiential|commercial|place|venue|destination|landmark|spot|city|memorial|village|isle|road|town|country'
        r'|street|park|square|coast|hill|mountain|beach|island|museaum|highland|coordinates|area|region|zone'
        r'|territory|locale|kilometre|west|east|north|south|district|vicinity|location|space|neighbor|landscape|environment|urban|rural'
        r'|suburban|metropolitan|geograph|topograph|terrain|size|boundaries|republic|valley'
        r'|africe|europe|americe|asia|united state|united kingdom|india|argentina|england|pakistan|germany|austria'
        r'|italy|sweden|china|japan|canada|brazil|france|spain|russia|holland|netherland'
        r'|scotland|poland|state|gold|cupper|silver',
        text, re.IGNORECASE)

    if match:
        categories.append("Geography and Places")

    return categories


def get_most_fitting_heuristic_keyword_shortened(text):
    """
    Get the most fitting heuristic keyword for shortened category type based on the input text.

    :param: text (str): Input text.
    :returns: str: The most fitting heuristic keyword.
    """

    categories_count = {
        "Culture and Arts": 0,
        "Health": 0,
        "History": 0,
        "Science": 0,
        "People": 0,
        "Religion": 0,
        "Society": 0,
        "Technology": 0,
        "Geography and Places": 0
    }

    # Culture and Arts
    match = re.search(
        r'(\b(tone|press|art|set|line|lines|form|key|act)\b)'
        r'|(scholar|news|media|journalism|reporter|broadcast|television|radio|print|social media|internet'
        r'|advertising|communication|philosophy|philosopher|ethics|metaphysics|thinker|thought|logic|existence'
        r'|mind|truth|knowledge|reality|marality|reason|rationality|perception|painting|sculpture|gallery'
        r'|artist|exhibition|canvas|brush|palette|composition|perspective|color|texture|style|medium'
        r'|technique|expression|realism|surrealism|modern|theater|drama|playwright|performance|play|rehearsal'
        r'|costume|lighting|props|curtain|film|movie|director|cinema|actor|actress|scene|script|editing'
        r'|soundtrack|visual|dialogue|critic|premiere|award|sequel|screenplay|music|composer|orchestra|guitar'
        r'|song|singer|concert|melody|rhythm|harmony|instrument|vocals|sound|album|band|recording'
        r'|lyrics|beat|chord|chorus|soloist|improvisation|stage|audience|novel|poetry|literature|author|book'
        r'|writing|fable|fables|saga|poem|protagonist|antagonist|prose|character|plot|setting|theme|symbolism'
        r'|metaphor|genre|narrative|symbol|serie|episode|star|magazine)',
        text, re.IGNORECASE)

    if match:
        categories_count["Culture and Arts"] += 1

    # Health
    match = re.search(
        r'\b(ice|sip|bar|win|run|lap|lose|wine|cup)\b'
        r'|(drink|beverage|coffee|cocktail|brewery|barista|alcohol|mix|glass|shake|stir|garnish|falvor'
        r'|liquor|champagn|soda|food|cuisine|cooking|restaurant|chef|recipe|dish|ingredients|flavors|taste|meal|appetizer'
        r'|dessert|spice|fresh|organic|medicine|medical|doctor|health|hospital|patient|treatment|disease'
        r'|symptom|diagonsis|injury|surgery|search|vaccine|therapy|clinic|specialist|laboratory|sport|soccer'
        r'|basketball|athlet|goal|captain|line-up|loan|league|winner|penalty|driver|tennis|football|baseball|player|competition|game|team|match|score|champion'
        r'|tournament|coach|train|success|victory|defeat|record|ball|overtime|field|court|referee|fans|stadium|season|club|draw|amateur)',
        text, re.IGNORECASE)

    if match:
        categories_count["Health"] += 1

    # History
    match = re.search(
        r'\b(war|era|date)\b'
        r'|(history|historical|ancient|timeline|event|period|government|empire|culture|tradition|heritage'
        r'|document|research|king|military|prince|century|troop|emperor|battle)',
        text, re.IGNORECASE)

    if match:
        categories_count["History"] += 1

    # Science
    match = re.search(
        r'\b(sea|air|sun|oak|rose|pet|ion|cat|map|base|egg|pine|bos|sus|eco|stem|lion|gene|fox|sand)\b|'
        r'(scholar|lake|aquatic|maritime|naval|beach|marine|environmental|sustainability|pollution'
        r'|sustainab|renewable|diodiversity|warming|greenhouse|deforestation|recycling|preservation|ocean'
        r'|footprint|impact|nature|landscape|environment|outdoors|wilderness|forest|river|wild|fauna'
        r'|biodiversity|flora|tree|flower|botany|gardening|horticulture|growth|leaves|roots|foliage'
        r'|nutrients|water|oxygen|carbon|pollination|reproduction|tulip|maple|cactus|orchid|animal'
        r'|wildlife|class|order|nest|vegetation|temperature|incub|predation|family|genus|zoology|creature|mammal'
        r'|reptile|amphibian|insect|predator|wood|garden'
        r'|prey|habitat|endangered|conservation|extinct|ecosystem|bio|adaption|camoflage|carnivore|herbivore'
        r'|omnivore|hibernation|noctural|diurnal|exotic|nesting|territory|zoo|fossil|dog|fish|bird|horse'
        r'|cow|pig|elephant|tiger|spider|griaffe|bear|rabbit|monkey|dolphin|shark|snake|penguin|koala|kanaroo'
        r'|wolf|owl|deer|camel|canis|felis|homo|humanities|philosophy|arts|literature|history|human|ethics'
        r'|language|civilization|cultur|exhibition|thinking|social science|sociology|anthropology|culture|psychology'
        r'|behavior|society|research|theory|data|analysis|socialogy|anthropologie|economics|political|gender'
        r'|globalization|power|social|geography|geographical|continent|geology|landform|topography|earth'
        r'|country|region|climate|terrain|resources|biome|latitude|longitude|population|urbanization|migration'
        r'|natural|plate|geo|cartography|mathematics|mathematical|algebra|geometry|calculus|equation'
        r'|mathematician|numbers|function|variable|theorem|proof|statistic|probability|matrix|derivative'
        r'|integral|polynom|graph|coordinate|prime|vector|chemical|compound|molecule|element|reaction|chemistry'
        r'|atom|chem|peroad|acid|solution|crystal|catalyst|electron|proton|neutron|isotope|redox'
        r'|oxid|reduction|polymer|eneregy|physics|physical|energy|quantum|mechanics|thermodynamics|particle'
        r'|matter|force|motion|velocity|mass|gravity|wave|light|magnetism|biology|biological|organism|ecology'
        r'|genetics|species|evolution|cell|dna|diversity|photosynthesis|food|chromosom|agriculture'
        r'|farming|crop|livestock|agronomy|harvest|farm|soil|irrigation|cultivation|plant|fertilizer|rural|star|moon|planet)',
        text, re.IGNORECASE)

    if match:
        categories_count["Science"] += 1

    # People
    match = re.search(
        r'\b(class|son)\b|'
        r'(education|educational|teaching|learning|school|curriculum|student|teach|know|skill|assessment'
        r'|instruction|university|biography|grade|life|memoir|autobiography|biographical|history|work|profile|person'
        r'|child|kid|adult|parent|friend|family|eductaion|career|archievements|legacy|influence|impact|timeline|events|challenges'
        r'|relationships|degree|worked|written|married|daughter|brother|sister|founder|father|mother)',
        text, re.IGNORECASE)

    if match:
        categories_count["People"] += 1

    # Religion
    match = re.search(
        r'\b(sin|hell)\b|'
        r'(religion|religious|faith|church|spirituality|belief|theology|god|prayer|spiritaulity|sacre|scripture'
        r'|mosque|temple|holy|ritual|salvation|heaven|priest|prophet|islam|hindu|christ|jude|jewish|buddhism'
        r'|bible|bishop)',
        text, re.IGNORECASE)

    if match:
        categories_count["Religion"] += 1

    # Society
    match = re.search(
        r'\b(law|rate)\b|'
        r'financ|banking|money|portfolio|stock|income|bank|expense|asset|liability|dividend|dollar'
        r'|euro|credit|corporate|company|entrepreneur|industry|enterprise|commerce|management|finance'
        r'|revenue|profit|invest|customer|gold|silver|billion|million|market|compet|strategy|employ|product|service|sustain|innovation'
        r'|risk|return|balance|liquidity|economics|economic|headquater|trade|economist|economy|business'
        r'|supply|demand|democrat|party|store'
        r'|gdp|inflation|unemployment|growth|tax|budget|investment|interest|finan|policy|regulation|captial'
        r'|consumer|production|politics|political|government|administration|leadership|democracy|legislation'
        r'|constitution|congress|president|senator|reprentative|cabinet|foreign|diplomacy|voter|lobbying'
        r'|debate|leisure|hobby|recreation|entertainment|pastime|relaxation|hobb|enjoyment|unwind|outdoor'
        r'|indoor|weekend|holiday|society|social|community|public|culture|norms|values|relationship|diversity'
        r'|equality|identity|class|gender|race|power|education|media|tradition|change|elect|organization',
        text, re.IGNORECASE)

    if match:
        categories_count["Society"] += 1

    # Technology
    match = re.search(
        r'\b(car|ship|port|jet|pi)\b|'
        r'(invention|inventor|gamma|parameter|plynomial|approximation|schema|graph|patent|discovery|creation|invent|prototype|advancement|breakthrough|development'
        r'|inventive|scirentific|creative|experiment|pioneering|revolution|transport|transportation|vehicle'
        r'|travel|railroad|aircraft|airship|airplane|logistic|shipping|road|freight|cargo|route|infrastructure|railway|delivery'
        r'|transit|traffic|engineering|engineer|mechanical|design|construction|optimization|efficiency|cost'
        r'|complexity|function|pressure|launch|performance|redundancy|analysis|quality|project|computer science|programming'
        r'|software|algorithm|comput|informatic|hardware|data|coding|system|network|artificial|machine|cyber'
        r'|operating|database|compiler|interface|experience|internet|cloud|program|robotic|graphic|info'
        r'|electronics|electronic|engine|gadget|device|technology|circuit|innovation|component|voltage|current'
        r'|resistance|capacitor|transistor|diode|micro|controller|conductor|amplifier|resistor|inductor'
        r'|digital|analyog|oscillator|phone|computer|laptop|mobile|andriod|touchscreen|apple|samsung|huawei)',
        text, re.IGNORECASE)

    if match:
        categories_count["Technology"] += 1

    # Geography and Places
    match = re.search(
        r'\b(river|km)\b|'
        r'cities|building|structure|architecture|edifice|construction|skyscraper|monument|design|foundation'
        r'|materials|facade|roof|wall|floor|mile|inland|window|interior|exterior|renovation|archtiect|engineer'
        r'|blueprint|resdiential|commercial|place|venue|destination|landmark|spot|city|memorial|village|isle|road|town|country'
        r'|street|park|square|coast|hill|mountain|beach|island|museaum|highland|coordinates|area|region|zone'
        r'|territory|locale|kilometre|west|east|north|south|district|vicinity|location|space|neighbor|landscape|environment|urban|rural'
        r'|suburban|metropolitan|geograph|topograph|terrain|size|boundaries|republic|valley'
        r'|africe|europe|americe|asia|united state|united kingdom|india|argentina|england|pakistan|germany|austria'
        r'|italy|sweden|china|japan|canada|brazil|france|spain|russia|holland|netherland'
        r'|scotland|poland|state|gold|cupper|silver',
        text, re.IGNORECASE)

    if match:
        categories_count["Geography and Places"] += 1

    # Find the category with the highest count
    max_category = max(categories_count, key=categories_count.get)

    return max_category


def get_keyword_heuristics_extended(text):
    """
    Get keyword-based heuristic labels for extended category type.

    :param: text (str): Input text.
    :returns: list: List of heuristic labels.
    """

    categories = []

    # Literature
    if re.search(
            r'(\b(tone)\b)'
            r'|(novel|poetry|literature|author|book|writing|fable|saga|poem|protagonist|antagonist|prose|character'
            r'|plot|setting|theme|symbolism|metaphor|genre|narrative|symbol)',
            text, re.IGNORECASE):
        categories.append("Literature")

    # Music
    if re.search(
            r'(music|composer|orchestra|guitar|song|singer|concert|melody|rhythm|harmony|instrument|vocals|sound'
            r'|album|artist|band|recording|lyrics|beat|genre|composition|chord|chorus|soloist|improvisation|stage|audience)',
            text, re.IGNORECASE):
        categories.append("Music")

    # Film
    if re.search(
            r'film|movie|director|cinema|actor|actress|screenplay|plot|scene|genre|script|editing|soundtrack|visual'
            r'|dialogue|critic|premiere|award|sequel',
            text, re.IGNORECASE):
        categories.append("Film")

    # Theater
    if re.search(
            r'(theater|drama|playwright|stage|actor|actress|performance|play|act|scene|director|audience|script'
            r'|rehearsal|costume|set|lighting|sound|props|curtain|plot)',
            text, re.IGNORECASE):
        categories.append("Theater")

    # Visual Arts
    if re.search(
            r'(art|painting|sculpture|gallery|artist|exhibition|canvas|brush|palette|composition|perspective|color|form'
            r'|line|texture|style|medium|technique|visual|expression|realism|surrealism|modern)',
            text, re.IGNORECASE):
        categories.append("Visual Arts")

    # Philosophy
    if re.search(
            r'(philosophy|philosopher|ethics|metaphysics|thinker|thought|logic|existence|mind|truth|knowledge|reality'
            r'|marality|reason|rationality|perception)',
            text, re.IGNORECASE):
        categories.append("Philosophy")

    # Media
    if re.search(
            r'(press|news|media|journalism|reporter|broadcast|television|radio|print|social media|internet|advertising'
            r'|communication)',
            text, re.IGNORECASE):
        categories.append("Media")

    # Sport
    if re.search(
            r'(sport|soccer|basketball|athlete|tennis|football|player|competition|game|team|match|score|win|lose'
            r'|champion|tournament|coach|train|victory|defeat|record|ball|field|court|referee|fans|stadium)',
            text, re.IGNORECASE):
        categories.append("Sport")

    # Medicine
    if re.search(
            r'(medicine|medical|doctor|health|hospital|patient|treatment|disease|symptom|diagonsis|surgery|search'
            r'|vaccine|therapy|clinic|specialist|laboratory)',
            text, re.IGNORECASE):
        categories.append("Medicine")

    # Food
    if re.search(
            r'(food|cuisine|cooking|restaurant|chef|recipe|dish|ingredients|flavors|taste|meal|appetizer|dessert|spice'
            r'|fresh|organic)',
            text, re.IGNORECASE):
        categories.append("Food")

    # Drink
    if re.search(
            r'drink|beverage|wine|coffee|cocktail|brewery|barista|alcohol|mix|glass|ice|shake|stir|garnish|sip|falvor'
            r'|liquor|recipe|bar|soda|taste',
            text, re.IGNORECASE):
        categories.append("Drink")

    # History
    if re.search(
            r'history|historical|war|ancient|timeline|era|event|period|government|emprie|culture|tradition|heritage'
            r'|document|research',
            text, re.IGNORECASE):
        categories.append("History")

    # Agriculture
    if re.search(
            r'agriculture|farming|crop|livestock|agronomy|harvest|farm|soil|irrigation|cultivation|plant|fertilizer|rural',
            text, re.IGNORECASE):
        categories.append("Agriculture")

    # Biology
    if re.search(
            r'biology|biological|organism|ecology|genetics|species|evolution|cell|dna|gene|diversity|eco|photosynthesis'
            r'|food|chromosom',
            text, re.IGNORECASE):
        categories.append("Biology")

    # Physics
    if re.search(
            r'physics|physical|energy|quantum|mechanics|thermodynamics|particle|matter|force|motion|velocity|mass'
            r'|gravity|wave|magnetism',
            text, re.IGNORECASE):
        categories.append("Physics")

    # Chemistry
    if re.search(
            r'chemistry|chemical|compound|molecule|element|reaction|atom|chem|peroad|acid|base|solution'
            r'|crystal|catalyst|electron|proton|neutron|ion|isotope|redox|oxid|reduction|polymer|eneregy',
            text, re.IGNORECASE):
        categories.append("Chemistry")

    # Mathematics
    if re.search(
            r'mathematics|mathematical|algebra|geometry|calculus|equation|mathematician|numbers|function|variable'
            r'|theorem|proof|statistic|probability|matrix|derivative|integral|polynom|graph|coordinate|prime|vector',
            text, re.IGNORECASE):
        categories.append("Mathematics")

    # Geography
    if re.search(
            r'geography|geographical|map|continent|geology|landform|topography|earth|country|region|climate|terrain'
            r'|resources|eco|biome|latitude|longitude|population|urbanization|migration|natural|plate|geo|cartography',
            text, re.IGNORECASE):
        categories.append("Geography")

    # Social Science
    if re.search(
            r'social science|sociology|anthropology|culture|psychology|behavior|society|research|theory|data|analysis'
            r'|socialogy|anthropologie|economics|political|gender|diversity|globalization|power|social',
            text, re.IGNORECASE):
        categories.append("Social Science")

    # Humanities
    if re.search(
            r'humanities|philosophy|arts|literature|history|culture|human|society|ethics|language|civilization|cultur'
            r'|thinking',
            text, re.IGNORECASE):
        categories.append("Humanities")

    # Animal
    if re.search(
            r'animal|species|wildlife|pet|class|order|family|genus|zoology|creature|mammal|reptile|amphibian|insect'
            r'|predator|prey|habitat|endangered|conservation|extinct|ecosystem|bio|migration|adaption|camoflage'
            r'|carnivore|herbivore|omnivore|hibernation|noctural|diurnal|exotic|nesting|territory|food|zoo|fossil'
            r'|evolution|dog|cat|fish|bird|horse|cow|pig|elephant|lion|tiger|griaffe|bear|rabbit|monkey|dolphin'
            r'|shark|snake|fox|penguin|koala|kanaroo|wolf|owl|deer|canis|felis|bos|sus|homo',
            text, re.IGNORECASE):
        categories.append("Animal")

    # Plant
    if re.search(
            r'plant|flora|tree|flower|botany|gardening|horticulture|growth|photosynthesis|leaves|roots|stem|foliage'
            r'|soil|nutrients|sun|water|oxygen|carbon|pollination|reproduction|eco|rose|oak|tulip|pine|maple|cactus|orchid',
            text, re.IGNORECASE):
        categories.append("Plant")

    # Nature
    if re.search(
            r'nature|natural|landscape|environment|ecosystem|outdoors|wilderness|forest|river|wild|flora|fauna|eco'
            r'|diodiversity',
            text, re.IGNORECASE):
        categories.append("Nature")

    # Environment
    if re.search(
            r'environmental|ecology|climate|sustainability|conservation|ecosystem|pollution|sustainab|renewable'
            r'|diodiversity|warming|resources|greenhouse|deforestation|water|recycling|preservation|air|carbon|ocean'
            r'|footprint|impact',
            text, re.IGNORECASE):
        categories.append("Environment")

    # Waters
    if re.search(r'water|ocean|river|lake|sea|aquatic|maritime|naval|environment|beach|sand|pollution|marine', text,
                 re.IGNORECASE):
        categories.append("Waters")

    # Biography
    if re.search(
            r'biography|life|memoir|autobiography|biographical|history|profile|lief|person|child|family|eductaion'
            r'|career|archievements|legacy|influence|impact|timeline|events|challenges|relationships',
            text, re.IGNORECASE):
        categories.append("Biography")

    # Education
    if re.search(
            r'education|educational|teaching|learning|school|curriculum|student|teach|class|know|skill|assessment'
            r'|instruction|university',
            text, re.IGNORECASE):
        categories.append("Education")

    # Religion
    if re.search(
            r'religion|religious|faith|church|spirituality|belief|theology|god|prayer|spiritaulity|sacre|scripture'
            r'|mosque|temple|holy|ritual|salvation|sin|heaven|hell|priest|prophet',
            text, re.IGNORECASE):
        categories.append("Religion")

    # Society
    if re.search(
            r'society|social|community|public|culture|norms|values|relationship|diversity|equality|identity|class'
            r'|gender|race|power|politics|education|media|tradition|change',
            text, re.IGNORECASE):
        categories.append("Society")

    # Leisure
    if re.search(
            r'leisure|hobby|recreation|entertainment|pastime|relaxation|hobb|enjoyment|unwind|outdoor|indoor|weekend'
            r'|holiday',
            text, re.IGNORECASE):
        categories.append("Leisure")

    # Politics
    if re.search(
            r'politics|political|government|policy|administration|leadership|democracy|legislation'
            r'|constitution|congress|president|senator|reprentative|cabinet|foreign|diplomacy|voter|public'
            r'|lobbying|debate|law',
            text, re.IGNORECASE):
        categories.append("Politics")

    # Economics
    if re.search(
            r'economics|economic|market|trade|economist|economy|business|supply|demand|gdp|inflation|unemployment'
            r'|growth|tax|budget|investment|interest|finan|policy|regulation|captial|consumer|production',
            text, re.IGNORECASE):
        categories.append("Economics")

    # Business
    if re.search(
            r'business|corporate|company|entrepreneur|industry|enterprise|commerce|management|finance|revenue'
            r'|profit|invest|customer|market|compet|strategy|leadership|employ|product|service|sustain'
            r'|innovation|growth|risk|return|balance|liquidity|supply',
            text, re.IGNORECASE):
        categories.append("Business")

    # Finance
    if re.search(
            r'finance|financ|banking|investment|money|portfolio|stock|market|economy|income|bank|expense|profit'
            r'|asset|liability|risk|interest|captial|dividend|rate|dollar|euro|inflation|tax|credit',
            text, re.IGNORECASE):
        categories.append("Finance")

    # Electronics
    if re.search(
            r'electronics|electronic|gadget|device|technology|circuit|innovation|component|voltage|current'
            r'|resistance|capacitor|transistor|diode|micro|controller|conductor|amplifier|resistor|inductor'
            r'|power|digital|analyog|oscillator',
            text, re.IGNORECASE):
        categories.append("Electronics")

    # Computer Science
    if re.search(
            r'computer science|programming|software|algorithm|comput|informatic|hardware|data|coding|system'
            r'|network|artificial|machine|cyber|operating|database|compiler|interface|experience|internet'
            r'|cloud|program|robotic|graphic|info',
            text, re.IGNORECASE):
        categories.append("Computer Science")

    # Engineering
    if re.search(
            r'engineering|engineer|technology|mechanical|innovation|design|construction|optimization|efficiency|cost'
            r'|complexity|function|performance|redundancy|analysis|quality|project',
            text, re.IGNORECASE):
        categories.append("Engineering")

    # Transport
    if re.search(
            r'transport|transportation|vehicle|travel|car|railroad|airplane|logistic|shipping|freight|cargo|route'
            r'|infrastructure|ship|railway|port|delivery|transit|traffic',
            text, re.IGNORECASE):
        categories.append("Transport")

    # Inventions
    if re.search(
            r'invention|innovation|inventor|patent|discovery|creation|invent|technology|prototype|advancement'
            r'|breakthrough|development|inventive|engineer|design|scirentific|creative|experiment|pioneering|revolution',
            text, re.IGNORECASE):
        categories.append("Inventions")

    # Areas
    if re.search(
            r'area|region|zone|territory|locale|district|vicinity|location|space|neighbor|landscape|environment'
            r'|urban|rural|suburban|metropolitan|geograph|topograph|terrain|size|boundaries',
            text, re.IGNORECASE):
        categories.append("Areas")

    # Places
    if re.search(
            r'place|location|site|venue|destination|landmark|spot|city|town|country|region|area|neighbor|street'
            r'|park|square|coast|mountain|river|beach|island|museaum|coordinates',
            text, re.IGNORECASE):
        categories.append("Places")

    # Buildings
    if re.search(
            r'building|structure|architecture|edifice|construction|skyscraper|monument|design|foundation|materials'
            r'|facade|roof|wall|floor|window|door|interior|exterior|renovation|archtiect|engineer|blueprint|urban'
            r'|resdiential|commercial',
            text, re.IGNORECASE):
        categories.append("Buildings")

    return categories


def count_years_in_text(text):
    """
    Count the occurrences of years in the input text.

    :param: text (str): Input text.
    :returns int: Number of occurrences of years.
    """

    year_pattern = r'\b(1[0-9]{3}|2[0-9]{3})\b'
    years = re.findall(year_pattern, text)

    return len(years)
