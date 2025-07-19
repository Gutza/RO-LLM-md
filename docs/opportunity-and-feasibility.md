Analysis and Recommendations for Fine-Tuning a Cost-Effective LLM for Romanian Markdown Correction
==================================================================================================

> **Purpose**: preliminary concept proofing
>
> **Date**: 2025-07-19
>
> **Model**: Google Gemini 2.5 Pro (Deep Research)
>
> **Prompt**: I want to train (fine-tune) an LLM to fix LLM-generated markdown in Romanian. All LLMs I used generate markdown which has at least two problems:
> 1. the markdown itself is wasteful (e.g. it uses both `#` and `**` to highlight titles)
> 2. even when generating Romanian text, the LLM uses English title capitalization, which is obnoxiously annoying.
>
> As you can see, my needs are not ambitious, so I'm happy with an older, smaller model. However, since this is a personal project/experiment, I need it to be cost-effective, too – both when using it, and when training it. I will probably kick off the first experiments with around 100 text samples, relatively short, probably 4-5 paragraphs long each. I will probably never need more than 500-1000 samples.
> Please help me select a good model for finetuning. I only want to use OpenAI or Google models. Please research this and present a short list of three options, with pros and cons.

# TL;DR
For your project of fine-tuning an LLM to correct Romanian markdown, **the optimal choice is OpenAI's `gpt-4.1-nano` model**, which offers **the lowest cost for both training and inference**. Its serverless, pay-as-you-go pricing is perfectly suited for a personal project with intermittent usage, as you only pay for what you use, with no fixed monthly fees.[1, 2] Should you require more power, `gpt-4o-mini` presents a slightly more capable yet still highly cost-effective alternative on the same platform.[1]

In contrast, Google's Gemini models via the Vertex AI platform are strongly discouraged; their enterprise-focused pricing includes mandatory, recurring hosting fees for a dedicated endpoint, leading to a prohibitive fixed cost of several hundred dollars per month regardless of usage.[3, 4] Therefore, the recommended strategy is to proceed with `gpt-4.1-nano` and focus on curating a high-quality training dataset of 50-100 "before and after" examples to teach the model the specific formatting rules you require.[5]

# Part 1: Foundational Concepts for Your Fine-Tuning Project
This report provides a detailed analysis and a set of actionable recommendations for selecting and fine-tuning a Large Language Model (LLM) for the specific task of correcting stylistic and formatting issues in Romanian markdown. The analysis is grounded in the primary constraints of a small-scale, personal project where cost-effectiveness for both the initial training and ongoing use is paramount. The evaluation is limited to models available through the OpenAI and Google Cloud platforms.

## The Rationale for Fine-Tuning: Modifying Behavior, Not Knowledge
A foundational understanding of what fine-tuning achieves is critical to project success. Fine-tuning is a process of specialization. It does not primarily serve to imbue a model with new factual knowledge; for that, techniques like Retrieval-Augmented Generation (RAG) are more appropriate.<sup>1</sup> Instead, fine-tuning excels at modifying a model's

*behavior*. This includes altering its style, tone, and, most relevant to this project, its output format.<sup>1</sup>

The specified task—correcting redundant markdown syntax (e.g., # \*\*Title\*\*) and enforcing Romanian capitalization rules instead of English ones—is a quintessential behavioral modification. The base models already possess the requisite "knowledge": they understand the Romanian language and the syntax of markdown. The problem lies in their failure to consistently apply a specific, nuanced set of stylistic rules. Fine-tuning is the ideal mechanism to correct these failures and align the model's output with a desired format.<sup>7</sup>

This strategic alignment between the task and the tool has significant implications. Because the goal is to adjust output patterns rather than teach complex new concepts, a high probability of success can be expected even with a relatively small, high-quality dataset. Furthermore, this focus on behavior means that smaller, less computationally intensive models are often perfectly sufficient. The objective is not to select the most knowledgeable model, but the one whose output behavior can be most efficiently and economically molded to the required specifications.

## Data Preparation: The Blueprint for Success
The performance of a fine-tuned model is a direct and uncompromising reflection of the quality and structure of its training data.<sup>4</sup> For both OpenAI and Google Vertex AI, the standard format for supervised fine-tuning is JSON Lines (JSONL), where each line is a self-contained JSON object representing a single training example.<sup>8</sup>

For this specific text-correction task, each JSONL object will contain a conversational turn. This consists of an input prompt (representing the problematic text) and an ideal completion (representing the corrected text). Within the API structure, this is typically formatted using user and assistant roles.<sup>8</sup>

A practical example tailored to this project would look as follows:
```JSON
{"messages":}
{"messages":}
```

In this structure, the user message provides the "before" state—the LLM-generated text with wasteful markdown and incorrect English-style title case. The assistant message provides the "after" state—the perfectly formatted text that the fine-tuned model should learn to produce.

Regarding dataset size, while more data can be beneficial, stylistic and formatting tasks often require fewer examples than knowledge-intensive ones. A starting point of 50 to 100 high-quality, representative examples is a well-established best practice and aligns with the plan to initiate experiments with around 100 samples.<sup>3</sup> As the project scales, expanding this to 500-1000 examples will further solidify the model's new behavior. It is also advisable to set aside a small portion of the curated data (e.g., 10-20%) as a

*validation set*. This data is not used for training but is used to periodically test the model's performance, helping to detect "overfitting"—a state where the model memorizes the training examples instead of learning the general rules.<sup>4</sup>

### The Hidden Cost of Language: The "Romanian Token Tax"
A critical factor that will profoundly impact the total cost of this project is the mechanics of tokenization. LLM pricing is not based on words or characters but on "tokens," which are common sequences of characters generated by an algorithm like Byte-Pair Encoding (BPE).<sup>16</sup> A common rule of thumb for English is that 100 tokens correspond to approximately 75 words.<sup>16</sup>

However, this efficiency does not hold for other languages. Research and empirical data show that tokenizers trained primarily on English-language corpora are significantly less efficient when processing text in other languages, particularly those containing diacritics or non-Latin scripts.<sup>19</sup> For example, the same content can require 2.6 times more tokens in Bulgarian or 3 times more in Arabic compared to English.<sup>20</sup> Even Romance languages like Spanish show a higher token-to-character ratio.<sup>16</sup>

This phenomenon creates a "Romanian Token Tax." The Romanian language uses several characters with diacritics (e.g., ă, â, î, ș, ț), which are often split into multiple tokens by standard tokenizers. Consequently, processing a Romanian document will consume substantially more tokens—and thus incur higher costs—than processing an English document of the same word count.

This is not a minor detail; it is a fundamental cost driver that must be explicitly modeled in any financial projection. To create realistic cost estimates for this project, a conservative "Romanian Penalty Factor" of 1.8x will be applied. This assumes that a given number of Romanian words will generate approximately 1.8 times as many tokens as the same number of English words. This adjustment transforms a generic price list into a tailored financial forecast that accounts for the specific linguistic context of the project, preventing significant underestimation of both training and operational costs.

# Part 2: A Comparative Analysis of Model Candidates
Based on the project's requirements for a small, cost-effective model available on the OpenAI or Google platforms, three primary candidates have been shortlisted for detailed analysis. Each is evaluated based on its technical capabilities, pricing structure, and the strategic implications of its use.

## Candidate 1: OpenAI gpt-4o-mini (The Balanced Performer)
The gpt-4o-mini model is a modern, efficient model designed to balance performance, speed, and cost. It possesses strong multilingual capabilities, including robust support for Romanian, making it a highly suitable base for this fine-tuning task.<sup>21</sup> Critically, it is available for supervised fine-tuning to all developers with a paid OpenAI API account, ensuring accessibility.<sup>22</sup>

The primary advantage of gpt-4o-mini lies in its straightforward and predictable pricing structure. The costs are entirely usage-based, with no hidden fees for infrastructure or hosting. This pay-as-you-go model is exceptionally well-suited for personal projects and experiments where usage may be intermittent. The costs are broken down as follows <sup>23</sup>:

* **Training Cost:** $3.00 per 1 million tokens processed.
* **Inference Cost (after fine-tuning):**
* $0.30 per 1 million input tokens.
* $1.20 per 1 million output tokens.

The absence of separate, recurring hosting fees is a decisive factor. Unlike some platforms that require a continuously running, billable server to host a fine-tuned model, OpenAI's approach is serverless. The fine-tuned model is hosted at no charge, and costs are incurred only when an API call is made.<sup>25</sup> This eliminates fixed overhead and ensures that the total cost scales directly with usage.

* **Pros:**
  * Excellent balance of modern capabilities and affordability.
  * Simple, transparent, and predictable pay-as-you-go pricing.
  * Strong official support and documentation for fine-tuning.
  * No recurring monthly hosting or compute fees for the deployed model.
* **Cons:**
  * While cost-effective, it is not the absolute cheapest option available from OpenAI.

### Candidate 2: OpenAI gpt-4.1-nano (The Ultra-Cost-Effective Option)
For projects where budget is the single most important constraint, gpt-4.1-nano presents a compelling alternative. Positioned as OpenAI's fastest and most economical model, it is optimized for low-latency, high-throughput tasks.<sup>27</sup> While it is the least powerful model in the GPT-4.1 series, the narrow and highly specific nature of the markdown correction task means that its reduced general reasoning capability is unlikely to be a significant impediment. The model's behavior can be effectively shaped with a sufficient number of high-quality training examples.

The pricing for gpt-4.1-nano is its most attractive feature, offering substantial savings over gpt-4o-mini <sup>23</sup>:

* **Training Cost:** $1.50 per 1 million tokens (a 50% reduction compared to gpt-4o-mini).
* **Inference Cost (after fine-tuning):**
* $0.20 per 1 million input tokens.
* $0.80 per 1 million output tokens (a 33% reduction compared to gpt-4o-mini).

This model shares the same advantageous serverless pricing architecture as gpt-4o-mini, with no hosting fees and a purely usage-based cost structure.

An important consideration that emerges from this analysis is the invalidity of the common assumption that older models are inherently cheaper. The pricing data reveals that the older gpt-3.5-turbo model is significantly more expensive to fine-tune, at $8.00 per million tokens, than both gpt-4o-mini ($3.00/1M) and gpt-4.1-nano ($1.50/1M).<sup>23</sup> This suggests a deliberate pricing strategy by OpenAI to incentivize developers to migrate from older architectures to its newer, more efficient models. For this project, it means that the most modern

*small* models are, counter-intuitively, the most cost-effective choices for fine-tuning.

* **Pros:**
  * The lowest absolute cost for both training and inference on the OpenAI platform.
  * Simple, predictable, and transparent pay-as-you-go pricing.
  * No fixed hosting fees, ensuring minimal cost for low-volume usage.
* **Cons:**
  * As the least powerful model, it may require a larger or higher-quality dataset to achieve the same level of performance as gpt-4o-mini.
  * There is a higher risk that it may struggle to perfectly capture the nuances of the task, potentially requiring more iteration.

### Candidate 3: Google gemini-2.5-flash via Vertex AI (The Complex Alternative)
Google's gemini-2.5-flash is a powerful and efficient model that supports supervised fine-tuning, positioning it as a direct competitor to OpenAI's smaller offerings.<sup>7</sup> However, the path to using a fine-tuned Gemini model is fundamentally different and significantly more complex, both technically and financially.

Crucially, fine-tuning for Gemini models is not available through the simple, developer-focused Gemini API (formerly Google AI Studio). Instead, it is an enterprise-level feature accessible only through the comprehensive Google Cloud Vertex AI platform.<sup>28</sup> This distinction introduces a multi-layered and opaque pricing structure that is poorly suited for small-scale, budget-conscious projects.

The total cost of ownership for a fine-tuned Gemini model on Vertex AI is composed of three separate and distinct components:

1. **Usage Fees (Tokens/Characters):** Like OpenAI, Vertex AI charges for the data processed by the model. This is often billed per 1,000 characters for older models or per 1 million tokens for newer ones, with rates that differ from the consumer-facing Gemini API.<sup>29</sup>
2. **Training Compute Costs:** Unlike OpenAI's all-inclusive training fee, Vertex AI bills for the underlying computational resources used during the fine-tuning process. This is an hourly charge for the specific virtual machine (VM) and graphics processing unit (GPU) configuration that runs the training job.<sup>32</sup> The duration of this job is not always predictable, making the final training cost variable and difficult to estimate accurately beforehand. There are community reports of unexpected high costs, with one user being charged over $250 for a small job with only 10 examples.<sup>33</sup>
3. **Inference Endpoint Hosting Costs:** This is the most significant and prohibitive cost component for a personal project. To use a fine-tuned model on Vertex AI, it must be deployed to a dedicated "Endpoint." This Endpoint is essentially a VM that runs continuously to serve prediction requests. It incurs an hourly fee, 24 hours a day, 7 days a week, regardless of whether any API calls are being made.<sup>30</sup> Even a minimal machine type can cost upwards of $0.75 per hour <sup>30</sup>, leading to a fixed monthly overhead of over $500 before a single token is even processed.

This infrastructure-centric pricing model is designed for enterprise applications with high, predictable traffic, where the cost of a dedicated server can be amortized over millions of requests. For a personal project with low or intermittent usage, this model creates a "Total Cost of Ownership (TCO) Trap." The seemingly competitive per-token rates are rendered irrelevant by the massive, fixed monthly cost of simply keeping the model available for use. The pricing model is fundamentally misaligned with the project's economic constraints.

* **Pros:**
  * A high-performance, modern model from a leading provider.
  * Deep integration with the broader Google Cloud Platform ecosystem.
* **Cons:**
  * Extremely complex, multi-part pricing structure.
  * Prohibitively high and unpredictable Total Cost of Ownership due to mandatory, recurring endpoint hosting fees.
  * Variable and opaque training costs based on compute time.
  * The platform's complexity is excessive for the defined task.

# Part 3: Detailed Cost Projection and Scenario Analysis
To translate the abstract pricing models into tangible figures, this section provides a detailed cost projection based on a set of standardized assumptions derived from the project description. This allows for a direct, apples-to-apples comparison of the financial implications of selecting each model.

## Modeling Assumptions
The following parameters form the basis of the cost calculations. They are designed to be realistic representations of the project's scale and linguistic context.

* **Average Document Size:** 400 words (based on the description of "4-5 paragraphs long each").
* **English Tokenization Rate:** 1 word is assumed to be approximately 1.33 tokens. This is the inverse of the common rule of thumb that 100 tokens equal 75 words.<sup>16</sup>
* **Romanian Penalty Factor:** 1.8x. As established in Part 1, this multiplier accounts for the lower efficiency of tokenizing Romanian text compared to English.<sup>20</sup>
* **Effective Romanian Tokenization Rate:** The total tokens per document are estimated as: 400 words×1.33 tokens/word×1.8=957.6≈960 tokens.
* **Training Epochs:** 4. This is a common default value for fine-tuning jobs and is recommended by OpenAI as a starting point.<sup>24</sup> The total training tokens are calculated as (Number of Samples × Tokens per Sample × Epochs).
* **Vertex AI Endpoint Cost:** A conservative estimate of $0.50 per hour for a minimal, continuously running machine type suitable for hosting the model. This translates to a fixed monthly cost of 0.50×24×30=$360.

## Projected Cost Comparison
The following table synthesizes the pricing data and modeling assumptions to project the costs across different stages and scales of the project. This provides a clear, data-driven view of the total cost of ownership for each candidate.

| Metric                             | OpenAI gpt-4.1-nano | OpenAI gpt-4o-mini | Google gemini-2.5-flash    |
| ---------------------------------- | ------------------- | ------------------ | -------------------------- |
| Training Cost / 1M Tokens          | $1.50<sup>23</sup>  | $3.00<sup>23</sup> | Variable (Compute Hour)<sup>32</sup> |
| Inference Input / 1M Tokens        | $0.20<sup>23</sup>  | $0.30<sup>23</sup> | ~$0.15 (Vertex AI rate)<sup>30</sup> |
| Inference Output / 1M Tokens       | $0.80<sup>23</sup>  | $1.20<sup>23</sup> | ~$0.60 (Vertex AI rate)<sup>30</sup> |
| --- Scenario Projections ---       |                     |                    |                            |
| Est. Training Cost (100 Samples)   | $0.58               | $1.15              | \$10 - $50+ (Est. compute)  |
| Est. Training Cost (1,000 Samples) | $5.76               | $11.52             | \$50 - $200+ (Est. compute) |
| Est. Inference Cost (per 1k docs)  | ~$0.96              | ~$1.44             | ~\$0.72 (Tokens only)       |
| Recurring Monthly Hosting Cost     | $0.00<sup>25</sup>  | $0.00<sup>25</sup> | ~\$360.00 (@ $0.50/hr)<sup>30</sup>   |
| Est. Total First-Year Cost\*       | ~$17.28             | ~\$28.80            | ~\$4,370.00+                |

*\*Assumes training on 1,000 samples and processing 1,000 documents per month. The inference cost for gemini-2.5-flash is included in the total but is negligible compared to the hosting fee.*

## Economic Analysis and Interpretation
The data presented in the table leads to an unequivocal conclusion. For a small-scale, personal project with intermittent usage, the serverless, pay-as-you-go pricing model offered by OpenAI is vastly superior from a financial standpoint.

The training costs for both OpenAI models are nominal, projected to be less than $12 even for a substantial dataset of 1,000 examples. In contrast, the training cost on Vertex AI is not only higher but also unpredictable, depending on the opaque duration of the compute job.

The most dramatic disparity, however, lies in the ongoing operational cost. The OpenAI models incur zero fixed costs. The total monthly expense is purely a function of usage, making it possible to run the model for just a few dollars per month if activity is low. The Google Vertex AI model, by contrast, imposes a fixed monthly hosting fee of approximately $360. This single cost component makes the solution over 100 times more expensive than the OpenAI alternatives on an annual basis for the projected usage level. Even if the per-token inference costs for Gemini were zero, the mandatory hosting fee would render it economically non-viable for this use case.

# Part 4: Strategic Recommendations and Implementation Roadmap
This final section synthesizes the preceding analysis into a set of clear, prioritized recommendations and provides a high-level roadmap to guide the initial phases of the project.

## Final Recommendations
The analysis overwhelmingly indicates that the OpenAI platform offers the most suitable and cost-effective solutions for this specific project.

* Primary Recommendation: OpenAI gpt-4.1-nano
  For a project where cost is the absolute primary driver, gpt-4.1-nano is the optimal choice. It provides the lowest possible price for both the one-time training and the ongoing inference, with no recurring fees. The potential trade-off in raw capability compared to more powerful models is likely negligible for the highly specific and repetitive task of format correction. Any performance gaps can almost certainly be overcome with a high-quality, well-curated training dataset.
* Secondary Recommendation: OpenAI gpt-4o-mini
  If initial experiments with gpt-4.1-nano do not yield satisfactory results, or if a more robust and generally capable base model is preferred for a marginal increase in cost, gpt-4o-mini is the ideal fallback option. It maintains the simple, predictable, and serverless pricing model that is critical for the project's financial viability while offering a step up in performance.
* Recommendation Against Google Vertex AI
  It is strongly recommended to avoid using Google's gemini-2.5-flash via the Vertex AI platform for this project. The pricing model, with its mandatory, recurring hourly fees for endpoint hosting and its variable, compute-based training costs, results in a total cost of ownership that is prohibitively expensive and fundamentally misaligned with the economic realities of a small-scale, personal experiment.

## High-Level Implementation Guide
The following steps outline a practical path forward for developing the fine-tuned model on the OpenAI platform.

1. **Data Curation and Augmentation:**
* **Seed Dataset:** Begin by manually creating a "golden" dataset of 50-100 high-quality examples. Each example should be a user/assistant pair demonstrating the correction of a specific markdown or capitalization error.
* **Data Augmentation:** To cost-effectively scale the dataset to the 500-1000 sample range, employ data augmentation. Take the "good" corrected texts from the seed dataset and use a powerful base model (e.g., the standard gpt-4o API) to paraphrase or rewrite them in various ways. After generating these variations, manually re-introduce the specific formatting errors that need to be fixed. This process creates a large and diverse training set from a small, high-quality core, teaching the model to apply the rules across different sentence structures and contexts.<sup>36</sup>
* **Inclusion of Correct Examples:** It is crucial to include examples in the training data that are *already correct*. In these cases, the user prompt and the assistant completion will be identical. This teaches the model to recognize correctly formatted text and leave it unchanged, preventing the model from developing a tendency to over-correct valid inputs.<sup>39</sup>

1. Execution via Python API:
   The fine-tuning process can be managed programmatically using the official openai Python library. The key steps are as follows 8:

* **Step 1: Upload Data.** Prepare the training data in the JSONL format and upload it to OpenAI's servers.
  ```Python
  from openai import OpenAI
  client = OpenAI()

  client.files.create(
    file=open("romanian_markdown_training.jsonl", "rb"),
    purpose="fine-tune"
  )
  ```

* **Step 2: Create Fine-Tuning Job.** Initiate the training job, specifying the ID of the uploaded file and the chosen base model ID (e.g., gpt-4.1-nano-2025-04-14 or gpt-4o-mini-2024-07-18).
  ```Python
  client.fine_tuning.jobs.create(
    training_file="FILE_ID_FROM_STEP_1",
    model="gpt-4.1-nano-2025-04-14"
  )
  ```

* **Step 3: Retrieve Model ID.** Once the job is complete, retrieve the ID of the newly created fine-tuned model.
  ```Python
  # List jobs to find the job ID and the resulting fine-tuned model ID
  # client.fine_tuning.jobs.list(limit=10)
  fine_tuned_model_id = "ft:gpt-4.1-nano..." # Example ID
  ```

* **Step 4: Make Inference Call.** Use the new model ID to make API calls for correction.
  ```Python
  completion = client.chat.completions.create(
    model=fine_tuned_model_id,
    messages=
  )
  print(completion.choices.message.content)
  ```

1. **Evaluation and Iteration:**

* **Monitor Metrics:** After the fine-tuning job completes, OpenAI provides metrics, including the training\_loss. For text correction and stylistic tasks, a steadily decreasing loss that stabilizes at a low value (ideally below 1.0) indicates that the model is successfully learning the desired patterns.<sup>4</sup>
* **Validate Performance:** Test the newly created model against the validation set (the examples it was not trained on). This is the most important step to confirm that the model has learned the general *rules* of formatting rather than simply memorizing the specific examples from the training data.
* **Iterate:** If the model's performance is not yet satisfactory, the most effective next step is to improve the quality, quantity, and diversity of the training data. Analyze the model's failures on the validation set to identify patterns of error, and then create new training examples that specifically address those shortcomings. This data-centric iteration is typically more effective and economical than immediately switching to a more powerful (and more expensive) base model.

# Works cited
1. Exploring the Intricacies of Fine-Tuning Through Loss Metrics - API, accessed July 19, 2025, [<u>https://community.openai.com/t/exploring-the-intricacies-of-fine-tuning-through-loss-metrics/820403</u>](https://community.openai.com/t/exploring-the-intricacies-of-fine-tuning-through-loss-metrics/820403)
2. Correcting wrong answers via fine-tuning - API - OpenAI Developer Community, accessed July 19, 2025, [<u>https://community.openai.com/t/correcting-wrong-answers-via-fine-tuning/554930</u>](https://community.openai.com/t/correcting-wrong-answers-via-fine-tuning/554930)
3. Conference Talk 18: Fine-Tuning OpenAI Models - Best Practices - Christian Mills, accessed July 19, 2025, [<u>https://christianjmills.com/posts/mastering-llms-course-notes/conference-talk-018/</u>](https://christianjmills.com/posts/mastering-llms-course-notes/conference-talk-018/)
4. How to Fine-Tune AI to Suit Your Writing Style - Novelcrafter, accessed July 19, 2025, [<u>https://www.novelcrafter.com/blog/fine-tuning-ai-for-authors</u>](https://www.novelcrafter.com/blog/fine-tuning-ai-for-authors)
5. Fine-tuning a Foundation Model for a Specific Task with Vertex AI - DEV Community, accessed July 19, 2025, [<u>https://dev.to/ifihan/fine-tuning-a-foundation-model-for-a-specific-task-with-vertex-ai-1mkc</u>](https://dev.to/ifihan/fine-tuning-a-foundation-model-for-a-specific-task-with-vertex-ai-1mkc)
6. How to Fine Tune GPT 3.5 Turbo - Medium, accessed July 19, 2025, [<u>https://medium.com/@alimusa2514/how-to-fine-tune-gpt-3-5-turbo-8253ed30fe09</u>](https://medium.com/@alimusa2514/how-to-fine-tune-gpt-3-5-turbo-8253ed30fe09)
7. About supervised fine-tuning for Gemini models | Generative AI on Vertex AI - Google Cloud, accessed July 19, 2025, [<u>https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning</u>](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning)
8. Fine-Tuning OpenAI's GPT-4: A Step-by-Step Guide - DataCamp, accessed July 19, 2025, [<u>https://www.datacamp.com/tutorial/fine-tuning-openais-gpt-4-step-by-step-guide</u>](https://www.datacamp.com/tutorial/fine-tuning-openais-gpt-4-step-by-step-guide)
9. Fine tuning Gemini with Google AI Studio - YouTube, accessed July 19, 2025, [<u>https://www.youtube.com/watch?v=-ja5TmYhQks\&pp=0gcJCfwAo7VqN5tD</u>](https://www.youtube.com/watch?v=-ja5TmYhQks\&pp=0gcJCfwAo7VqN5tD)
10. A Step-by-Step Guide to Fine-Tuning Gemini for Question Answering | by E. Huizenga | Google Cloud - Medium, accessed July 19, 2025, [<u>https://medium.com/google-cloud/a-step-by-step-guide-to-fine-tuning-gemini-for-question-answering-8b3fb117dbbf</u>](https://medium.com/google-cloud/a-step-by-step-guide-to-fine-tuning-gemini-for-question-answering-8b3fb117dbbf)
11. How to fine-tune Vertex AI LLMs with Labelbox, accessed July 19, 2025, [<u>https://labelbox.com/guides/how-to-fine-tune-vertex-ai-models-with-labelbox/</u>](https://labelbox.com/guides/how-to-fine-tune-vertex-ai-models-with-labelbox/)
12. How to fine-tune chat models | OpenAI Cookbook, accessed July 19, 2025, [<u>https://cookbook.openai.com/examples/how\_to\_finetune\_chat\_models</u>](https://cookbook.openai.com/examples/how_to_finetune_chat_models)
13. Introduction to tuning | Generative AI on Vertex AI - Google Cloud, accessed July 19, 2025, [<u>https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models</u>](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-models)
14. Finetuning Noob : Guidelines and Best Practices? - API - OpenAI Developer Community, accessed July 19, 2025, [<u>https://community.openai.com/t/finetuning-noob-guidelines-and-best-practices/404778</u>](https://community.openai.com/t/finetuning-noob-guidelines-and-best-practices/404778)
15. What is overfitting in LLM fine-tuning? - Talbot West, accessed July 19, 2025, [<u>https://talbotwest.com/ai-insights/what-is-overfitting-in-llm</u>](https://talbotwest.com/ai-insights/what-is-overfitting-in-llm)
16. What are tokens and how to count them? - OpenAI Help Center, accessed July 19, 2025, [<u>https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them</u>](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
17. OpenAI Token Calculator - QuizRise.com, accessed July 19, 2025, [<u>https://www.quizrise.com/token-counter</u>](https://www.quizrise.com/token-counter)
18. LLM Tokenization - Hundred Blocks, accessed July 19, 2025, [<u>https://hundredblocks.github.io/transcription\_demo/</u>](https://hundredblocks.github.io/transcription_demo/)
19. The tokenization pipeline - Hugging Face, accessed July 19, 2025, [<u>https://huggingface.co/docs/tokenizers/pipeline</u>](https://huggingface.co/docs/tokenizers/pipeline)
20. Language Model Tokenizers Introduce Unfairness Between Languages - arXiv, accessed July 19, 2025, [<u>https://arxiv.org/pdf/2305.15425</u>](https://arxiv.org/pdf/2305.15425)
21. List of languages supported by ChatGPT - Botpress, accessed July 19, 2025, [<u>https://botpress.com/blog/list-of-languages-supported-by-chatgpt</u>](https://botpress.com/blog/list-of-languages-supported-by-chatgpt)
22. Fine-tuning now available for GPT-4o - OpenAI, accessed July 19, 2025, [<u>https://openai.com/index/gpt-4o-fine-tuning/</u>](https://openai.com/index/gpt-4o-fine-tuning/)
23. Pricing - OpenAI API, accessed July 19, 2025, [<u>https://platform.openai.com/docs/pricing</u>](https://platform.openai.com/docs/pricing)
24. How Much Does It Cost to Fine-Tune GPT-4o mini? - FinetuneDB, accessed July 19, 2025, [<u>https://finetunedb.com/blog/how-much-does-it-cost-to-finetune-gpt-4o-mini/</u>](https://finetunedb.com/blog/how-much-does-it-cost-to-finetune-gpt-4o-mini/)
25. Does hosting fine-tuned models cost? - API - OpenAI Developer Community, accessed July 19, 2025, [<u>https://community.openai.com/t/does-hosting-fine-tuned-models-cost/501734</u>](https://community.openai.com/t/does-hosting-fine-tuned-models-cost/501734)
26. Fine-Tuning AI Models: Comparing the Costs of OpenAI vs Azure OpenAI | Vlad Iliescu, accessed July 19, 2025, [<u>https://vladiliescu.net/finetuning-costs-openai-vs-azure-openai/</u>](https://vladiliescu.net/finetuning-costs-openai-vs-azure-openai/)
27. API Pricing - OpenAI, accessed July 19, 2025, [<u>https://openai.com/api/pricing/</u>](https://openai.com/api/pricing/)
28. Fine-tuning with the Gemini API | Google AI for Developers, accessed July 19, 2025, [<u>https://ai.google.dev/gemini-api/docs/model-tuning</u>](https://ai.google.dev/gemini-api/docs/model-tuning)
29. Google Cloud Vertex AI Pricing Review 2025: Plans & Costs - Tekpon, accessed July 19, 2025, [<u>https://tekpon.com/software/google-cloud-vertex-ai/pricing/</u>](https://tekpon.com/software/google-cloud-vertex-ai/pricing/)
30. Re: Confused about pricing differences between Vertex AI and Google AI Studio, accessed July 19, 2025, [<u>https://www.googlecloudcommunity.com/gc/AI-ML/Confused-about-pricing-differences-between-Vertex-AI-and-Google/m-p/887859</u>](https://www.googlecloudcommunity.com/gc/AI-ML/Confused-about-pricing-differences-between-Vertex-AI-and-Google/m-p/887859)
31. I'm so confused about vertexai costs : r/googlecloud - Reddit, accessed July 19, 2025, [<u>https://www.reddit.com/r/googlecloud/comments/1ikqmle/im\_so\_confused\_about\_vertexai\_costs/</u>](https://www.reddit.com/r/googlecloud/comments/1ikqmle/im_so_confused_about_vertexai_costs/)
32. AI/ML Pricing on Google Cloud Platform - DEV Community, accessed July 19, 2025, [<u>https://dev.to/ddeveloperr/understanding-google-cloud-platform-pricing-gcp-pricing-59h4</u>](https://dev.to/ddeveloperr/understanding-google-cloud-platform-pricing-gcp-pricing-59h4)
33. Vertex AI Fine Tuning Pricing - Google Cloud Community, accessed July 19, 2025, [<u>https://www.googlecloudcommunity.com/gc/AI-ML/Vertex-AI-Fine-Tuning-Pricing/m-p/747286/highlight/true</u>](https://www.googlecloudcommunity.com/gc/AI-ML/Vertex-AI-Fine-Tuning-Pricing/m-p/747286/highlight/true)
34. Confused about pricing differences between Vertex AI and Google AI Studio - especially deployment costs : r/googlecloud - Reddit, accessed July 19, 2025, [<u>https://www.reddit.com/r/googlecloud/comments/1jfk2jb/confused\_about\_pricing\_differences\_between\_vertex/</u>](https://www.reddit.com/r/googlecloud/comments/1jfk2jb/confused_about_pricing_differences_between_vertex/)
35. Deploy a model to an endpoint | Vertex AI - Google Cloud, accessed July 19, 2025, [<u>https://cloud.google.com/vertex-ai/docs/general/deployment</u>](https://cloud.google.com/vertex-ai/docs/general/deployment)
36. (PDF) Text Data Augmentation for Large Language Models: A Comprehensive Survey of Methods, Challenges, and Opportunities - ResearchGate, accessed July 19, 2025, [<u>https://www.researchgate.net/publication/388634279\_Text\_Data\_Augmentation\_for\_Large\_Language\_Models\_A\_Comprehensive\_Survey\_of\_Methods\_Challenges\_and\_Opportunities</u>](https://www.researchgate.net/publication/388634279_Text_Data_Augmentation_for_Large_Language_Models_A_Comprehensive_Survey_of_Methods_Challenges_and_Opportunities)
37. Text Data Augmentation for Large Language Models: A Comprehensive Survey of Methods, Challenges, and Opportunities - arXiv, accessed July 19, 2025, [<u>https://arxiv.org/html/2501.18845v1</u>](https://arxiv.org/html/2501.18845v1)
38. Empowering Large Language Models for Textual Data Augmentation - arXiv, accessed July 19, 2025, [<u>https://arxiv.org/html/2404.17642v1</u>](https://arxiv.org/html/2404.17642v1)
39. Fine-tuning with negative samples, possible? - Prompting - OpenAI Developer Community, accessed July 19, 2025, [<u>https://community.openai.com/t/fine-tuning-with-negative-samples-possible/885002</u>](https://community.openai.com/t/fine-tuning-with-negative-samples-possible/885002)
40. How can I fine-tune a language model with negative examples using SFTTrainer?, accessed July 19, 2025, [<u>https://stackoverflow.com/questions/78216628/how-can-i-fine-tune-a-language-model-with-negative-examples-using-sfttrainer</u>](https://stackoverflow.com/questions/78216628/how-can-i-fine-tune-a-language-model-with-negative-examples-using-sfttrainer)
41. Fine-Tuning GPT-3 Using the OpenAI API and Python - DataCamp, accessed July 19, 2025, [<u>https://www.datacamp.com/tutorial/fine-tuning-gpt-3-using-the-open-ai-api-and-python</u>](https://www.datacamp.com/tutorial/fine-tuning-gpt-3-using-the-open-ai-api-and-python)