# AUTOSAR RAG Eval Dataset v1.0

## Motivation

This dataset is a gold-standard RAG evaluation benchmark for
AUTOSAR technical documentation. It was built with a three-stage
pipeline (over-generate → validate → finalize) using open-weight
models only, following the methodology from:
- RAGalyst (Gao et al., Nov 2025)
- Judge's Verdict (NVIDIA, Oct 2025)
- TREC 2024 RAG Track nugget methodology

## Composition

- Total samples: **365**
- Train / Dev / Test: **219 / 73 / 73**
- Rejected during finalization: **2635**

### Synthesizer distribution

- multi_hop_specific: 193 (52.9%)
- multi_hop_abstract: 172 (47.1%)

### Persona distribution

- tier1_developer: 126 (34.5%)
- application_sw_engineer: 86 (23.6%)
- autosar_integrator: 86 (23.6%)
- functional_safety_engineer: 67 (18.4%)

### Source document distribution

- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_ATS_CommunicationViaBus.pdf`: 55 (15.1%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_PRS_E2EProtocol.pdf`: 33 (9.0%)
- `/home/olj3kor/praveen/Autosar_docs_2/Adaptive Platform Machine Configuration.pdf`: 26 (7.1%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_FunctionalSafetyMeasures.pdf`: 24 (6.6%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_BSWDistributionGuide.pdf`: 24 (6.6%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_PRS_SOMEIPServiceDiscoveryProtocol.pdf`: 22 (6.0%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_PRS_LogAndTraceProtocol.pdf`: 21 (5.8%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_ErrorDescription.pdf`: 21 (5.8%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_ATS_CommunicationFlexRay.pdf`: 17 (4.7%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_ASWS_TransformerGeneral.pdf`: 17 (4.7%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_VFB.pdf`: 14 (3.8%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_ARAComAPI.pdf`: 14 (3.8%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_ApplicationLevelErrorHandling.pdf`: 13 (3.6%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_PRS_TimeSyncProtocol.pdf`: 12 (3.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_PRS_SOMEIPProtocol.pdf`: 11 (3.0%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_NVDataHandling.pdf`: 9 (2.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_UtilizationOfCryptoServices.pdf`: 9 (2.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_AIUserGuide.pdf`: 8 (2.2%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_Features.pdf`: 7 (1.9%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_ParallelProcessingGuidelines.pdf`: 7 (1.9%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_SafetyUseCase.pdf`: 6 (1.6%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_CPP14Guidelines.pdf`: 6 (1.6%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_COM.pdf`: 6 (1.6%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_ProjectObjectives.pdf`: 5 (1.4%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_CDDDesignAndIntegrationGuideline.pdf`: 5 (1.4%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_SafetyOverview.pdf`: 5 (1.4%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_AdaptivePlatformInterfacesGuidelines.pdf`: 5 (1.4%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_BSWGeneral.pdf`: 5 (1.4%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_PlatformDesign.pdf`: 4 (1.1%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_AIChassis.pdf`: 3 (0.8%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_SafetyExtensions.pdf`: 3 (0.8%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_FlashTest.pdf`: 3 (0.8%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_AIBodyAndComfort.pdf`: 3 (0.8%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_FreeRunningTimer.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_InterruptHandlingExplanation.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_LogAndTrace.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_MacroEncapsulationofInterpolationCalls.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_FeatureModelExchangeFormat.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_BSWModuleDescriptionTemplate.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_IPDUMultiplexer.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_LIN.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_LayeredSoftwareArchitecture.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_SecurityManagement.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_FlexRay.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_SensorInterfaces.pdf`: 2 (0.5%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_Methodology.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_E2E.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_ModeManagementGuide.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_PRS_NetworkManagementProtocol.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_ExecutionManagement.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_CommunicationManagement.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_CAN.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_SOMEIPProtocol.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_CoreTest.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_EXP_AIPowertrain.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_InteroperabilityOfAutosarTools.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_RS_StandardizationTemplate.pdf`: 1 (0.3%)
- `/home/olj3kor/praveen/Autosar_docs_2/AUTOSAR_SRS_ICUDriver.pdf`: 1 (0.3%)

### Score statistics

- Mean faithfulness: **1.000**
- Mean answer relevance: **0.986**

## Collection process

Candidates were generated from the source PDFs using
Qwen/Qwen2.5-72B-Instruct-AWQ with controlled personas and a
split question/answer prompt (answer prompt returns NOT_ANSWERABLE
when the context is insufficient).

Each candidate was scored by Qwen/Qwen3-30B-A3B-Instruct-2507
on four metrics: answerability, faithfulness, answer relevance,
and question specificity. Thresholds applied in this dataset:

- answerability == 1 (binary)
- question_specificity == 1 (binary)
- faithfulness >= 0.85
- answer_relevance >= 0.8
- no structural rule failures (see `shared/validators.py`)

## Known limitations

- Judge agreement with human experts has NOT yet been calibrated
  on this domain. Run the `human_review_queue.csv` spot-check to
  compute Cohen's κ between the judge and 2-3 domain SMEs before
  using scores for publication.
- Synthetic questions, so they may not match the distribution of
  real user queries. Augment with real queries in production use.
- No decontamination check was performed against common RAG
  benchmarks (MS MARCO, TREC, HotpotQA). If the source PDFs are
  public, overlap with pretraining corpora is possible.

## Recommended use

- **train** split: for prompt engineering / RAG tuning
- **dev** split: for iteration during development
- **test** split: held out, used rarely for final reporting