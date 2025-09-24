# Agentic AI with Orchestrator-Agent Trust: A Modular Visual Classification Framework with Trust-Aware Orchestration and RAG-Based Reasoning

## Article
* **Journal**: pending
* **Title**: [Agentic AI with Orchestrator-Agent Trust: A Modular Visual Classification Framework with Trust-Aware Orchestration and RAG-Based Reasoning](https://arxiv.org/abs/2507.10571)
* **DOI**: [https://doi.org/10.48550/arXiv.2507.10571](https://doi.org/10.48550/arXiv.2507.10571)

## Authors
* **Adj. Asst. Prof. Konstantinos I. Roumeliotis | University of Peloponnese**
* **Dr. Ranjan Sapkota | Cornell University**
* **Prof. Manoj Karkee | Cornell University**
* **Prof. Nikolaos D. Tselikas | University of Peloponnese**

## Abstract
Modern Artificial Intelligence (AI) increasingly relies on multi-agent architectures that blend visual and language understanding. Yet, a pressing challenge remains: How can we trust these agents especially in zero-shot settings with no fine-tuning? We introduce a novel modular Agentic AI visual classification framework that integrates generalist multimodal agents with a non-visual reasoning orchestrator and a Retrieval-Augmented Generation (RAG)  module. Applied to apple leaf disease diagnosis, we benchmark three configurations: (I) zero-shot with confidence-based orchestration, (II) fine-tuned agents with improved performance, and (III) trust-calibrated orchestration enhanced by CLIP-based image retrieval and re-evaluation loops. Using confidence calibration metrics (ECE, OCR, CCC), the orchestrator modulates trust across agents. Our results demonstrate a 77.94\% accuracy improvement in the zero-shot setting using trust-aware orchestration and RAG, achieving 85.63\% overall. GPT-4o showed better calibration, while Qwen-2.5-VL displayed overconfidence. Furthermore, image-RAG grounded predictions with visually similar cases, enabling correction of agent overconfidence via iterative re-evaluation. The proposed system separates perception (vision agents) from meta-reasoning (orchestrator), enabling scalable and interpretable multi-agent AI. This blueprint illustrates how Agentic AI can deliver trustworthy, modular, and transparent reasoning, and is extensible to diagnostics, biology, and other trust-critical domains. In doing so, we highlight Agentic AI not just as an architecture but as a paradigm for building reliable multi-agent intelligence. All models, prompts, results, and system components including the complete software source code are openly released to support reproducibility, transparency, and community benchmarking at our [Github](https://github.com/Applied-AI-Research-Lab/Orchestrator-Agent-Trust) page.

## Keywords
agentic ai, orchestrator agent trust, trust orchestration, visual classification, retrieval augmented reasoning
