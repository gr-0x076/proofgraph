# ProofGraph: Dedicated TinyFish Web Retrieval Evaluation Report
**Execution Mode:** `LIVE TINYFISH API`  
**Timestamp:** `2026-11-11 09:37:35 UTC`  
**Sample Size:** `N=10 claims` across history, science, geography, and current events.  

## 📊 Executive Summary Metrics
- **Evidence Retrieval Coverage:** `100.0%` (10/10 claims retrieved verified web passages)
- **End-to-End Classification Accuracy:** `50.0%` (5/10 matched ground truth)
- **Total Evidence Passages Ingested:** `30` passages
- **Total Chromium Pages Rendered & Fetched:** `24` full pages via TinyFetch
- **Total Wall Time:** `115.73s` (Average: `11.57s/claim`)  

## 🌐 Retrieved Domain Distribution
| Domain | Frequency | Authority Category |
| :--- | :---: | :--- |
| `en.wikipedia.org` | 5 | Reference / Wiki |
| `www.nasa.gov` | 2 | Government / Science |
| `www.reddit.com` | 2 | News / General |
| `pmc.ncbi.nlm.nih.gov` | 2 | Government / Science |
| `www.nobelprize.org` | 2 | News / General |
| `www.history.com` | 1 | News / General |
| `www.toureiffel.paris` | 1 | News / General |
| `www.walksofitaly.com` | 1 | News / General |
| `www.facebook.com` | 1 | News / General |
| `www.thermoworks.com` | 1 | News / General |
| `www.acs.org` | 1 | News / General |
| `www.skyatnightmagazine.com` | 1 | News / General |
| `www.advancedsciencenews.com` | 1 | News / General |
| `medium.com` | 1 | News / General |
| `www.audacy.com` | 1 | News / General |
| `www.gq.com.au` | 1 | News / General |
| `www.vishvasnews.com` | 1 | News / General |
| `www.instagram.com` | 1 | News / General |
| `www.claymath.org` | 1 | News / General |
| `www.usatoday.com` | 1 | News / General |
| `www.ncsl.org` | 1 | News / General |
| `www.peoriaelections.gov` | 1 | Government / Science |

## 🔬 Claim-by-Claim Verification Audit Trail
| Claim | Category | Expected | Predicted | Conf | Passages | Latency | Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Neil Armstrong walked on the Moon on July 20,... | Science & History | `SUPPORTED` | `SUPPORTED` | 95.0% | 3 | 6.52s | ✅ |
| The Eiffel Tower was constructed in Rome, Ita... | Geography & Architecture | `CONTRADICTED` | `CONTRADICTED` | 81.0% | 3 | 11.64s | ✅ |
| Water boils at 50 degrees Celsius at sea leve... | Physics & Chemistry | `CONTRADICTED` | `CONTRADICTED` | 95.0% | 3 | 14.84s | ✅ |
| Alexander Fleming discovered penicillin in 19... | Medicine & Biology | `SUPPORTED` | `CONTRADICTED` | 59.6% | 3 | 12.09s | ❌ |
| The Great Wall of China is visible from the M... | Astronomy Myth | `CONTRADICTED` | `SUPPORTED` | 95.0% | 3 | 10.24s | ❌ |
| Albert Einstein was awarded the 1921 Nobel Pr... | History of Science | `SUPPORTED` | `SUPPORTED` | 95.0% | 3 | 4.3s | ✅ |
| Trump Administration claimed songwriter Billi... | Political Reporting | `CONTRADICTED` | `SUPPORTED` | 64.0% | 3 | 17.99s | ❌ |
| UNESCO declared the Nadar community as the mo... | Social Media Rumor | `CONTRADICTED` | `CONTRADICTED` | 95.0% | 3 | 8.12s | ✅ |
| Quantum computers have officially proven that... | Theoretical Computer Science | `INSUFFICIENT_EVIDENCE` | `CONTRADICTED` | 50.5% | 3 | 11.26s | ❌ |
| All USA mail-in ballots arriving after electi... | Electoral Law | `CONTRADICTED` | `SUPPORTED` | 95.0% | 3 | 18.68s | ❌ |

## 🔍 Protocol Disclosure & Non-Comparability Note
> [!NOTE]
> This benchmark evaluates autonomous web discovery and Chromium page rendering via TinyFish Search and Fetch APIs.
> It is evaluated on an open-web claim set and is **strictly separated** from ProofGraph's candidate-pool AVeriTeC benchmark table to ensure transparent and unpolluted reporting.