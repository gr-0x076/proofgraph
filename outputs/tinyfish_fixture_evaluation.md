# ProofGraph: Dedicated TinyFish Web Retrieval Evaluation Report
**Execution Mode:** `FIXTURE REPLAY (Offline Diagnostic)`
**Timestamp:** `2026-11-11 16:24:38 UTC`
**Sample Size:** `N=10 claims` across history, science, geography, and current events.

## 📊 Executive Summary Metrics
- **Search Coverage:** `100.0%` (10/10 claims returned at least one candidate; this does not establish relevance or source quality)
- **End-to-End Classification Accuracy:** `90.0%` (9/10 matched ground truth)
- **Total Evidence Passages Ingested:** `10` passages
- **Total Chromium Pages Rendered & Fetched:** `10` full pages via TinyFetch
- **Total Wall Time:** `0.92s` (Average: `0.09s/claim`)

## 🌐 Retrieved Domain Distribution
| Domain | Frequency | Authority Category |
| :--- | :---: | :--- |
| `www.nasa.gov` | 2 | Government / Science |
| `www.nobelprize.org` | 2 | News / General |
| `en.wikipedia.org` | 1 | Reference / Wiki |
| `www.britannica.com` | 1 | News / General |
| `www.factcheck.org` | 1 | News / General |
| `en.unesco.org` | 1 | Government / Science |
| `www.claymath.org` | 1 | News / General |
| `www.ncsl.org` | 1 | News / General |

## 🔬 Claim-by-Claim Verification Audit Trail
| Claim | Category | Expected | Predicted | Conf | Passages | Latency | Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Neil Armstrong walked on the Moon on July 20,... | Science & History | `SUPPORTED` | `SUPPORTED` | 95.0% | 1 | 0.0s | ✅ |
| The Eiffel Tower was constructed in Rome, Ita... | Geography & Architecture | `CONTRADICTED` | `CONTRADICTED` | 95.0% | 1 | 0.0s | ✅ |
| Water boils at 50 degrees Celsius at sea leve... | Physics & Chemistry | `CONTRADICTED` | `CONTRADICTED` | 95.0% | 1 | 0.0s | ✅ |
| Alexander Fleming discovered penicillin in 19... | Medicine & Biology | `SUPPORTED` | `SUPPORTED` | 95.0% | 1 | 0.0s | ✅ |
| The Great Wall of China is visible from the M... | Astronomy Myth | `CONTRADICTED` | `CONTRADICTED` | 95.0% | 1 | 0.0s | ✅ |
| Albert Einstein was awarded the 1921 Nobel Pr... | History of Science | `SUPPORTED` | `SUPPORTED` | 95.0% | 1 | 0.0s | ✅ |
| Trump Administration claimed songwriter Billi... | Political Reporting | `CONTRADICTED` | `CONTRADICTED` | 95.0% | 1 | 0.0s | ✅ |
| UNESCO declared the Nadar community as the mo... | Social Media Rumor | `CONTRADICTED` | `CONTRADICTED` | 91.0% | 1 | 0.0s | ✅ |
| Quantum computers have officially proven that... | Theoretical Computer Science | `INSUFFICIENT_EVIDENCE` | `CONTRADICTED` | 93.7% | 1 | 0.0s | ❌ |
| All USA mail-in ballots arriving after electi... | Electoral Law | `CONTRADICTED` | `CONTRADICTED` | 91.1% | 1 | 0.0s | ✅ |

## Retrieved Source URLs
The records below make individual results auditable; source presence is not an endorsement of source reliability.
| Claim | Title | Domain | Retrieval | HTTP | URL |
| :--- | :--- | :--- | :--- | :---: | :--- |
| Neil Armstrong walked on the Moon on July 20,... | Apollo 11 Mission Overview \| NASA | www.nasa.gov | tinyfish_fetch | — | [https://www.nasa.gov/mission_pages/apollo/apollo-11.html](https://www.nasa.gov/mission_pages/apollo/apollo-11.html) |
| The Eiffel Tower was constructed in Rome, Ita... | Eiffel Tower - Wikipedia | en.wikipedia.org | tinyfish_fetch | — | [https://en.wikipedia.org/wiki/Eiffel_Tower](https://en.wikipedia.org/wiki/Eiffel_Tower) |
| Water boils at 50 degrees Celsius at sea leve... | Boiling Point \| Britannica | www.britannica.com | tinyfish_fetch | — | [https://www.britannica.com/science/boiling-point](https://www.britannica.com/science/boiling-point) |
| Alexander Fleming discovered penicillin in 19... | Alexander Fleming Biographical \| NobelPrize.org | www.nobelprize.org | tinyfish_fetch | — | [https://www.nobelprize.org/prizes/medicine/1945/fleming/biographical/](https://www.nobelprize.org/prizes/medicine/1945/fleming/biographical/) |
| The Great Wall of China is visible from the M... | China's Wall Less and More Viewable from Space \| NASA | www.nasa.gov | tinyfish_fetch | — | [https://www.nasa.gov/vision/earth/terrestrialsearch/great_wall.html](https://www.nasa.gov/vision/earth/terrestrialsearch/great_wall.html) |
| Albert Einstein was awarded the 1921 Nobel Pr... | The Nobel Prize in Physics 1921 \| NobelPrize.org | www.nobelprize.org | tinyfish_fetch | — | [https://www.nobelprize.org/prizes/physics/1921/summary/](https://www.nobelprize.org/prizes/physics/1921/summary/) |
| Trump Administration claimed songwriter Billi... | Fact Check: Billie Eilish Document \| FactCheck.org | www.factcheck.org | tinyfish_fetch | — | [https://www.factcheck.org/2020/10/billie-eilish-trump-administration-claim/](https://www.factcheck.org/2020/10/billie-eilish-trump-administration-claim/) |
| UNESCO declared the Nadar community as the mo... | UNESCO Clarification on Ancient Race Claims | en.unesco.org | tinyfish_fetch | — | [https://en.unesco.org/news/clarification-on-ancient-race-claims](https://en.unesco.org/news/clarification-on-ancient-race-claims) |
| Quantum computers have officially proven that... | P vs NP Problem \| Clay Mathematics Institute | www.claymath.org | tinyfish_fetch | — | [https://www.claymath.org/millennium-problems/p-vs-np-problem](https://www.claymath.org/millennium-problems/p-vs-np-problem) |
| All USA mail-in ballots arriving after electi... | Receipt and Postmark Deadlines for Absentee/Mail Ballots \| NCSL | www.ncsl.org | tinyfish_fetch | — | [https://www.ncsl.org/elections-and-campaigns/receipt-and-postmark-deadlines-for-ballots](https://www.ncsl.org/elections-and-campaigns/receipt-and-postmark-deadlines-for-ballots) |

## 🔍 Protocol Disclosure & Non-Comparability Note
> [!NOTE]
> This benchmark evaluates autonomous web discovery and Chromium page rendering via TinyFish Search and Fetch APIs.
> It is evaluated on an open-web claim set and is **strictly separated** from ProofGraph's candidate-pool AVeriTeC benchmark table to ensure transparent and unpolluted reporting.
