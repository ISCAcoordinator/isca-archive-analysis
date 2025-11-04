from bertopic.representation import TextGeneration
from bertopic.representation import BaseRepresentation
import huggingface_hub
import transformers
from torch import bfloat16


def define_speech_category_prompt() -> str:
    # System prompt describes information given to all conversations
    system_prompt = """
    <s>[INST]
    <<SYS>>
    You are a helpful, respectful and honest expert in speech science and speech technology acting as an assistantfor labeling topics.
    <</SYS>>
    """

    # Example prompt demonstrating the output we are looking for
    example_prompt = """
    I have a topic that contains the following documents:

    "
    Speech science is an expansive interdisciplinary field that delves into the complexities of human speech, encompassing the biological, psychological, and linguistic mechanisms involved in the production, transmission, and perception of spoken language.
    This field draws on diverse disciplines including anatomy, to explore the structure of the vocal apparatus; neuroscience, to understand the brain's role in language processing; and linguistics, to investigate the systematic and structural aspects of language itself.
    At conferences like Interspeech, researchers from around the world convene to share their latest findings on topics ranging from the minute biomechanics of vocal fold vibration to the cognitive processes underlying speech comprehension and production.
    This collaborative environment fosters a deeper understanding of how speech functions across different languages and populations, and it highlights the translational potential of speech science in areas such as speech pathology, language education, and cognitive rehabilitation, aiming to address a wide array of communication disorders and challenges.

    "

    The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

    Based on the information about the topic above, please assign the label "Speech Science", "Speech Technology" or "Equally Speech Science and Speech Technology" to this topic.
    Make sure you to only return the label and nothing more.

    [/INST] Speech Science
    """

    # Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
    main_prompt = """
    [INST]
    I have a topic that contains the following documents:

    "
    [DOCUMENTS]
    "
    The topic is described by the following keywords: '[KEYWORDS]'.

    Based on the information about the topic above, please create a short label of this topic.
    Make sure you to only return the label and nothing more.
    [/INST]
    """

    prompt = system_prompt + example_prompt + main_prompt


def define_topic_label_prompt() -> str:
    # System prompt describes information given to all conversations
    system_prompt = """
    <s>[INST]
    <<SYS>>
    You are a helpful, respectful and honest expert in speech science and speech technology acting as an assistantfor labeling topics.
    <</SYS>>
    """

    # Example prompt demonstrating the output we are looking for
    example_prompt = """
    I have a topic that contains the following documents:

    "
    Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
    Meat, but especially beef, is the word food in terms of emissions.
    Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.
    "

    The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

    Based on the information about the topic above, please create a short label of this topic.
    Make sure you to only return the label and nothing more.

    [/INST] Environmental impacts of eating meat
    """

    # Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
    main_prompt = """
    [INST]
    I have a topic that contains the following documents:

    "
    [DOCUMENTS]
    "
    The topic is described by the following keywords: '[KEYWORDS]'.

    Based on the information about the topic above, please create a short label of this topic.
    Make sure you to only return the label and nothing more.
    [/INST]
    """
    prompt = system_prompt + example_prompt + main_prompt


def configure_llama(token: str, model_id="meta-llama/Llama-2-7b-chat-hf") -> BaseRepresentation:

    # Login to huggingface hub
    huggingface_hub.login(token=token)

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit quantization
        bnb_4bit_quant_type="nf4",  # Normalized float 4
        bnb_4bit_use_double_quant=True,  # Second quantization after the first
        bnb_4bit_compute_dtype=bfloat16,  # Computation type
    )

    # Prepare Llama 2
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    # Our text generator
    generator = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1,
    )

    return TextGeneration(generator, prompt=define_topic_label_prompt())
