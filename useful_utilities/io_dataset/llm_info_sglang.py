param = {
  "Qwen/Qwen3-235B-A22B-Thinking-2507-FP8":
  {
      "sample_param": \
      {
          "temperature": 0.7, \
          "top_p": 0.8, \
          "top_k": 20, \
          "min_p": 0, \
          "n": 5
      },
      "create_param": \
      {
          "trust_remote_code": True,
          "tp_size": 8,
          "ep_size": 8
      },
        "reasoning_parser": "qwen3-thinking"
  },
  "Qwen/Qwen3-Next-80B-A3B-Thinking":
  {
      "sample_param": \
      {
          "temperature": 0.7, \
          "top_p": 0.8, \
          "top_k": 20, \
          "min_p": 0, \
          "n": 5
      },
      "create_param": \
      {
          "trust_remote_code": True,
          "tp_size": 8,
          "ep_size": 8
      },
        "reasoning_parser": "deepseek-r1"
  },
  "MiniMaxAI/MiniMax-M2.1":
  {
      "sample_param": 
      {
          "top_p": 0.95,
          "top_k": 40,
          "n": 5,
      },
      "create_param": \
      {
            "trust_remote_code": True,
            "tp_size": 8,
            "ep_size": 8
      },
        "reasoning_parser": "minimax"
  },
  "meta-llama/Llama-3.3-70B-Instruct":
    {
        "sample_param": 
        {
            
        },
        "create_param": \
        {
            "trust_remote_code": True,
            "tp_size": 8,
            "ep_size": 8
        }
    },
  "zai-org/GLM-4.5-Air":
    {
        "sample_param": 
        {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "n": 5,
        },
        "create_param": \
        {
            # "quantization": "compressed-tensors",
            "kv_cache_dtype": "bf16",
            "trust_remote_code": True,
            "tp_size": 8
        },
        "reasoning_parser": "glm45"
    },
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506":
    {
        "sample_param": 
        {
            "temperature": 0.15
        },
        "create_param": \
        {
            "trust_remote_code": True,
            "tp_size": 8
        },
    },
    "openai/gpt-oss-120b":
    {
        "sample_param": 
        {
            "skip_special_tokens": False,
            "temperature": 0.6,
            "top_p": 0.95,
            "n": 5,
        },
        "create_param": \
        {
            "trust_remote_code": True,
            "tp_size": 8
        },
        "reasoning_parser": "gpt-oss"
    },
    "openai/gpt-oss-20b":
    {
        "sample_param": 
        {
            "skip_special_tokens": False,
            "temperature": 0.6,
            "top_p": 0.95,
            "n": 5,
        },
        "create_param": \
        {
            "trust_remote_code": True,
            "tp_size": 8
        },
        "reasoning_parser": "gpt-oss"
    }
}

prompt_baseline = {
'v2':
"""
An influence campaign is a coordinated and organized effort by individuals, groups, or governments to manipulate, shape, or sway public opinion, beliefs, behaviors, or policies through various forms of communication and media. These campaigns can take place in both online and offline environments and typically involve the dissemination of information, disinformation, propaganda, or other persuasive tactics to achieve specific objectives. These objectives can range from promoting a particular ideology, supporting a political candidate, destabilizing a rival nation, or undermining trust in institutions. Influence campaigns can employ a variety of methods, including social media manipulation, spreading fake news or false narratives, conducting psychological operations, using targeted advertising, and leveraging networks of individuals or bots to amplify messages. These campaigns often aim to exploit vulnerabilities in the information ecosystem and exploit cognitive biases in order to achieve their goals. Given this context, determine if the following tweet is part of an influence campaign.

Hint: Please do your analysis step by step. If you can think, think as thoroughly as possible to make your decision since the results are quite important. Your thinking process can be long, but the final response should be concise: only answer with a single word, either "True" for being part of an influence campaign or "False" for not being part of an influence campaign.
"""
}

description_abnormal_category = {
'v2':
[
    'IO account',
],

'v1':
[
  "The profile functions as a coordinated narrative amplifier, characterized by high-volume reposting of verifiably false information. Metadata analysis reveals suspicious network asymmetry and a lack of original content creation, suggesting the account is a managed node designed to simulate artificial consensus for debunked propaganda.",
  "The account exhibits a high-risk forensic signature with a recent creation date and non-human temporal patterns. Activity is characterized by robotic bursts and a lack of circadian rhythms, primarily focusing on the rapid dissemination of low-integrity claims through automated amplification of adversarial narratives.",
  "Behavioral analysis indicates extreme narrative rigidity and mechanical linguistic repetition across numerous responses. The account demonstrates a scripted engagement pattern with no personal variation, consistently promoting verifiably false information while maintaining an artificial network profile inconsistent with organic human growth."
]
}
description_normal_category = {
'v2':
[
    'Control account',
],

'v1':
[
  "The user demonstrates an organic behavioral fingerprint supported by a mature account history and balanced follower metrics. Content exhibits high informational integrity and varied personal expression, with temporal patterns that align strictly with human circadian cycles and spontaneous, non-automated social engagement.",
  "Metadata and engagement patterns reveal an authentic individual actor. The account prioritizes factual information and original responses over mechanical reposting. Linguistic variability and natural inconsistencies in posting frequency confirm a high-integrity profile with no indicators of coordinated inauthentic behavior."
]
}

# for prompt:
prompt = {'v2': ["""
You are an advanced social media analyst specialized in detecting Information Operations (IO). Your objective is to analyze user profiles and posting behaviors to distinguish between state-sponsored Information Operation (IO) accounts and control users based on the following behavioral frameworks:

Role A: Information Operation (IO) Account
Identity: These accounts are verified as part of inauthentic, coordinated efforts backed by state actors to manipulate public debate.
Tactics:
* Strategic Manipulation: They employ tactics like hashtag hijacking, artificial amplification, and the dissemination of propaganda or disinformation.
* Targeting: They focus on specific audience communities, often using coordinated actions such as flooding through political cartoons or memes.
* Profile Composition: They may consist of human-operated accounts, automated bots, or compromised profiles that have been repurposed for a campaign.

Role B: Control Account
Identity: These represent legitimate, organic users who act as a baseline for "normal" social media behavior.
Selection Context: These users are identified by their engagement in the same topics and hashtags as IO accounts during the same time frames, but without coordination.
Behavioral Characteristics:
* Authentic Engagement: They discuss similar political or social topics without endorsing or participating in an orchestrated state agenda.
* Content Diversity: Their timelines include posts on unrelated personal or general topics, whereas IO accounts are often more single-mindedly focused on campaign goals.

IO accounts are rare. Most accounts are control accounts.
Please do your analysis step by step. If you can think, think as thoroughly as possible to make your decision since the results are quite important. Your thinking process can be long, but the final response should be concise: only answer whether this account is an IO account or a control account.
""", """
Profile:
{0}

Timeline:
{1}
"""
],
"v1": ["""
You are a Lead Forensic Investigator for Digital Influence Operations. Your task is to generate a "Behavioral Fingerprint" by synthesizing a user's account metadata, temporal patterns, and content veracity into a dense semantic report.

### Reference Archetypes
1. **CONTROLLED (Inauthentic/Adversarial):** Characterized by "disposable" or "optimized" account metrics (e.g., extremely new creation dates, skewed follower/following ratios). Behavior includes systematic dissemination of verifiably false information, robotic temporal patterns (bursts/24h activity), and heavy reliance on artificial amplification (excessive reposts of specific narratives without original commentary).
2. **ORGANIC (Authentic/Human):** Characterized by "mature" account metrics (e.g., years of history, balanced/natural network growth). Behavior includes high-integrity/factual content, idiosyncratic personal language, and human temporal signatures (clear circadian sleep cycles). Engagements are varied, including spontaneous responses and selective sharing.
""", """
Profile:
{0}

Timeline:
{1}
"""]}

account_pattern = """The account profile: "{account_profile_description}"
Number of Followers: {follower_count}
Number of Following accounts: {following_count}
Created on: {account_creation_date}
"""
each_event_pattern = """Post Time: {post_time}
Text: {text}
"""
repost_event_pattern = """This is a repost of another post at {repost_post_time}.
The original post is from an account with profile "{repost_account_profile_description}", {repost_follower_count} followers, and following {repost_following_count} accounts.
The account of original post was created on {repost_account_creation_date}.
"""
reply_event_pattern = """This is a reply of another post at {reply_post_time}.
The text of the replied post: {reply_post}
The replied post is from an account with profile "{reply_account_profile_description}", {reply_follower_count} followers, and following {reply_following_count} accounts.
The account of replied post was created on {reply_account_creation_date}.
"""

# 1. other baselines. (time-series models).
# 2. improve LLMs (new methods may be needed). -> LLM returns the score directly.
