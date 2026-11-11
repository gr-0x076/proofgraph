# ProofGraph: Dedicated TinyFish Web Retrieval Evaluation Report
**Execution Mode:** `LIVE TINYFISH API`
**Timestamp:** `2026-11-11 10:38:14 UTC`
**Sample Size:** `N=10 claims` across history, science, geography, and current events.

## 📊 Executive Summary Metrics
- **Search Coverage:** `100.0%` (10/10 claims returned at least one candidate; this does not establish relevance or source quality)
- **End-to-End Classification Accuracy:** `50.0%` (5/10 matched ground truth)
- **Total Evidence Passages Ingested:** `26` passages
- **Total Chromium Pages Rendered & Fetched:** `14` full pages via TinyFetch
- **Total Wall Time:** `113.31s` (Average: `11.33s/claim`)

## 🌐 Retrieved Domain Distribution
| Domain | Frequency | Authority Category |
| :--- | :---: | :--- |
| `www.nasa.gov` | 2 | Government / Science |
| `www.facebook.com` | 2 | News / General |
| `en.wikipedia.org` | 2 | Reference / Wiki |
| `www.nobelprize.org` | 2 | News / General |
| `www.history.com` | 1 | News / General |
| `www.toureiffel.paris` | 1 | News / General |
| `www.journeystoitaly.com` | 1 | News / General |
| `www.walksofitaly.com` | 1 | News / General |
| `www.thermoworks.com` | 1 | News / General |
| `www.pewresearch.org` | 1 | News / General |
| `www.acs.org` | 1 | News / General |
| `pmc.ncbi.nlm.nih.gov` | 1 | Government / Science |
| `testbook.com` | 1 | News / General |
| `www.gq.com.au` | 1 | News / General |
| `nypost.com` | 1 | News / General |
| `www.vishvasnews.com` | 1 | News / General |
| `www.instagram.com` | 1 | News / General |
| `www.reddit.com` | 1 | News / General |
| `qc.ascsn.net` | 1 | News / General |
| `www.usnews.com` | 1 | News / General |
| `www.ncsl.org` | 1 | News / General |
| `statesunited.org` | 1 | News / General |

## 🔬 Claim-by-Claim Verification Audit Trail
| Claim | Category | Expected | Predicted | Conf | Passages | Latency | Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Neil Armstrong walked on the Moon on July 20,... | Science & History | `SUPPORTED` | `SUPPORTED` | 95.0% | 3 | 18.08s | ✅ |
| The Eiffel Tower was constructed in Rome, Ita... | Geography & Architecture | `CONTRADICTED` | `CONTRADICTED` | 95.0% | 3 | 16.9s | ✅ |
| Water boils at 50 degrees Celsius at sea leve... | Physics & Chemistry | `CONTRADICTED` | `CONTRADICTED` | 95.0% | 3 | 18.62s | ✅ |
| Alexander Fleming discovered penicillin in 19... | Medicine & Biology | `SUPPORTED` | `CONTRADICTED` | 59.6% | 3 | 11.2s | ❌ |
| The Great Wall of China is visible from the M... | Astronomy Myth | `CONTRADICTED` | `SUPPORTED` | 95.0% | 1 | 5.5s | ❌ |
| Albert Einstein was awarded the 1921 Nobel Pr... | History of Science | `SUPPORTED` | `SUPPORTED` | 95.0% | 3 | 4.04s | ✅ |
| Trump Administration claimed songwriter Billi... | Political Reporting | `CONTRADICTED` | `SUPPORTED` | 54.0% | 2 | 4.88s | ❌ |
| UNESCO declared the Nadar community as the mo... | Social Media Rumor | `CONTRADICTED` | `CONTRADICTED` | 95.0% | 3 | 5.07s | ✅ |
| Quantum computers have officially proven that... | Theoretical Computer Science | `INSUFFICIENT_EVIDENCE` | `CONTRADICTED` | 95.0% | 2 | 10.9s | ❌ |
| All USA mail-in ballots arriving after electi... | Electoral Law | `CONTRADICTED` | `SUPPORTED` | 95.0% | 3 | 18.08s | ❌ |

## Retrieved Source URLs
The records below make individual results auditable; source presence is not an endorsement of source reliability.
| Claim | Title | Domain | Retrieval | HTTP | URL |
| :--- | :--- | :--- | :--- | :---: | :--- |
| Neil Armstrong walked on the Moon on July 20,... | Apollo 11 | www.nasa.gov | tinyfish_search | — | [https://www.nasa.gov/mission/apollo-11/](https://www.nasa.gov/mission/apollo-11/) |
| Neil Armstrong walked on the Moon on July 20,... | One giant leap for mankind History was made on July 20 ... | www.facebook.com | tinyfish_search | — | [https://www.facebook.com/NASAHistoryOffice/posts/one-giant-leap-for-mankindhistory-was-made-on-july-20-1969-when-apollo-11s-neil-/1342679034709867/](https://www.facebook.com/NASAHistoryOffice/posts/one-giant-leap-for-mankindhistory-was-made-on-july-20-1969-when-apollo-11s-neil-/1342679034709867/) |
| Neil Armstrong walked on the Moon on July 20,... | Neil Armstrong walks on moon \| July 20, 1969 | www.history.com | tinyfish_search | — | [https://www.history.com/this-day-in-history/july-20/armstrong-walks-on-moon](https://www.history.com/this-day-in-history/july-20/armstrong-walks-on-moon) |
| The Eiffel Tower was constructed in Rome, Ita... | Eiffel Tower history, architecture, design & construction | www.toureiffel.paris | tinyfish_search | — | [https://www.toureiffel.paris/en/the-monument/history](https://www.toureiffel.paris/en/the-monument/history) |
| The Eiffel Tower was constructed in Rome, Ita... | Did You Know the Eiffel Tower Was Meant to Be Temporary? | www.journeystoitaly.com | tinyfish_search | — | [https://www.journeystoitaly.com/did-you-know-the-eiffel-tower-was-meant-to-be-temporary/](https://www.journeystoitaly.com/did-you-know-the-eiffel-tower-was-meant-to-be-temporary/) |
| The Eiffel Tower was constructed in Rome, Ita... | 7 Things You Never Knew About The Eiffel Tower ... | www.walksofitaly.com | tinyfish_search | — | [https://www.walksofitaly.com/blog/art-culture/eiffel-tower-information](https://www.walksofitaly.com/blog/art-culture/eiffel-tower-information) |
| Water boils at 50 degrees Celsius at sea leve... | The Boiling Point of Water Calculator | www.thermoworks.com | tinyfish_search | — | [https://www.thermoworks.com/blogs/learning-center/boiling-point-calculator?srsltid=AYCX9vKhRQuOy3TQcg4rWPtZv0iavcMjFJ4SIJucHZH3WdIqABlqCG8i](https://www.thermoworks.com/blogs/learning-center/boiling-point-calculator?srsltid=AYCX9vKhRQuOy3TQcg4rWPtZv0iavcMjFJ4SIJucHZH3WdIqABlqCG8i) |
| Water boils at 50 degrees Celsius at sea leve... | What temperature does water boil at? | www.facebook.com | tinyfish_search | — | [https://www.facebook.com/nshss/posts/what-temperature-does-water-boil-at/812297357612939/](https://www.facebook.com/nshss/posts/what-temperature-does-water-boil-at/812297357612939/) |
| Water boils at 50 degrees Celsius at sea leve... | Does water's boiling point change with altitude? Americans ... | www.pewresearch.org | tinyfish_search | — | [https://www.pewresearch.org/short-reads/2015/09/14/does-waters-boiling-point-change-with-altitude-americans-arent-sure/](https://www.pewresearch.org/short-reads/2015/09/14/does-waters-boiling-point-change-with-altitude-americans-arent-sure/) |
| Alexander Fleming discovered penicillin in 19... | Alexander Fleming Discovery and Development of Penicillin - Landmark | www.acs.org | tinyfish_fetch | — | [https://www.acs.org/education/whatischemistry/landmarks/flemingpenicillin.html](https://www.acs.org/education/whatischemistry/landmarks/flemingpenicillin.html) |
| Alexander Fleming discovered penicillin in 19... | Alexander Fleming - Wikipedia | en.wikipedia.org | tinyfish_fetch | — | [https://en.wikipedia.org/wiki/Alexander_Fleming](https://en.wikipedia.org/wiki/Alexander_Fleming) |
| Alexander Fleming discovered penicillin in 19... | The Discovery of Penicillin—New Insights After More Than 75 Years of Clinical Use - PMC | pmc.ncbi.nlm.nih.gov | tinyfish_fetch | — | [https://pmc.ncbi.nlm.nih.gov/articles/PMC5403050/](https://pmc.ncbi.nlm.nih.gov/articles/PMC5403050/) |
| The Great Wall of China is visible from the M... | Great Wall - NASA | www.nasa.gov | tinyfish_fetch | — | [https://www.nasa.gov/image-article/great-wall/](https://www.nasa.gov/image-article/great-wall/) |
| Albert Einstein was awarded the 1921 Nobel Pr... | Nobel Prize in Physics 1921 | www.nobelprize.org | tinyfish_fetch | — | [https://www.nobelprize.org/prizes/physics/1921/summary/](https://www.nobelprize.org/prizes/physics/1921/summary/) |
| Albert Einstein was awarded the 1921 Nobel Pr... | Nobel Prize in Physics 1921 | www.nobelprize.org | tinyfish_fetch | — | [https://www.nobelprize.org/prizes/physics/1921/einstein/facts/](https://www.nobelprize.org/prizes/physics/1921/einstein/facts/) |
| Albert Einstein was awarded the 1921 Nobel Pr... | [Solved] Albert Einstein was awarded the Nobel Prize for ______ in 19 | testbook.com | tinyfish_fetch | — | [https://testbook.com/question-answer/albert-einstein-was-awarded-the-nobel-prize-for-__--61d490ec75d83190f48f1f47](https://testbook.com/question-answer/albert-einstein-was-awarded-the-nobel-prize-for-__--61d490ec75d83190f48f1f47) |
| Trump Administration claimed songwriter Billi... | Billie Eilish Is “Destroying Our Country” Claims Leaked Trump Administration Document | www.gq.com.au | tinyfish_fetch | — | [https://www.gq.com.au/culture/entertainment/billie-eilish-is-destroying-our-country-claims-leaked-trump-administration-document/news-story/7debcb8921413b915d1033629e962721?nk=ae5f0d1899247ff1d1889caf4658f536-1794393382](https://www.gq.com.au/culture/entertainment/billie-eilish-is-destroying-our-country-claims-leaked-trump-administration-document/news-story/7debcb8921413b915d1033629e962721?nk=ae5f0d1899247ff1d1889caf4658f536-1794393382) |
| Trump Administration claimed songwriter Billi... | Washington Post wrongly claims Trump officials said Billie Eilish is ‘destroying our country’ | nypost.com | tinyfish_fetch | — | [https://nypost.com/2020/10/30/washington-post-wrongly-claims-trump-officials-criticized-billie-eilish/](https://nypost.com/2020/10/30/washington-post-wrongly-claims-trump-officials-criticized-billie-eilish/) |
| UNESCO declared the Nadar community as the mo... | Fact Check: UNESCO did not declare Nadar community as ancient race in the world, viral claim is fake | www.vishvasnews.com | tinyfish_fetch | — | [https://www.vishvasnews.com/english/viral/fact-check-unesco-did-not-declare-nadar-community-as-ancient-race-in-the-world-viral-claim-is-fake/](https://www.vishvasnews.com/english/viral/fact-check-unesco-did-not-declare-nadar-community-as-ancient-race-in-the-world-viral-claim-is-fake/) |
| UNESCO declared the Nadar community as the mo... | Nadar (caste) - Wikipedia | en.wikipedia.org | tinyfish_fetch | — | [https://en.wikipedia.org/wiki/Nadar_(caste)](https://en.wikipedia.org/wiki/Nadar_(caste)) |
| UNESCO declared the Nadar community as the mo... | Unesco Nadar Community Ancient Race Declaration | www.instagram.com | tinyfish_fetch | — | [https://www.instagram.com/popular/unesco-nadar-community-ancient-race-declaration/](https://www.instagram.com/popular/unesco-nadar-community-ancient-race-declaration/) |
| Quantum computers have officially proven that... | Will QC prove P=NP, or will it erase the need to prove it completely? : r/QuantumComputing | www.reddit.com | tinyfish_fetch | — | [https://www.reddit.com/r/QuantumComputing/comments/njkznw/will_qc_prove_pnp_or_will_it_erase_the_need_to/?solution=8ca3323a72db3c2f8ca3323a72db3c2f&js_challenge=1&jsc_token=2824be10929bdc604753c70a67a1c3310481b2dea63520db31cfedd5811239e3&jsc_orig_r=](https://www.reddit.com/r/QuantumComputing/comments/njkznw/will_qc_prove_pnp_or_will_it_erase_the_need_to/?solution=8ca3323a72db3c2f8ca3323a72db3c2f&js_challenge=1&jsc_token=2824be10929bdc604753c70a67a1c3310481b2dea63520db31cfedd5811239e3&jsc_orig_r=) |
| Quantum computers have officially proven that... | Computer science: P/NP space & quantum computing — Quantum Computing Applications in Nuclear Physics | qc.ascsn.net | tinyfish_fetch | — | [https://qc.ascsn.net/intro-qc/computer_science.html](https://qc.ascsn.net/intro-qc/computer_science.html) |
| All USA mail-in ballots arriving after electi... | Many States Count Mail Ballots That Arrive After Election ... | www.usnews.com | tinyfish_search | — | [https://www.usnews.com/news/best-states/ohio/articles/2026-03-20/many-states-count-mail-ballots-that-arrive-after-election-day-those-grace-periods-could-go-away](https://www.usnews.com/news/best-states/ohio/articles/2026-03-20/many-states-count-mail-ballots-that-arrive-after-election-day-those-grace-periods-could-go-away) |
| All USA mail-in ballots arriving after electi... | Table 7: When States Mail Out Absentee/Mail Ballots | www.ncsl.org | tinyfish_search | — | [https://www.ncsl.org/elections-and-campaigns/table-7-when-states-mail-out-absentee-mail-ballots](https://www.ncsl.org/elections-and-campaigns/table-7-when-states-mail-out-absentee-mail-ballots) |
| All USA mail-in ballots arriving after electi... | Sharing the Facts About Mail Voting | statesunited.org | tinyfish_search | — | [https://statesunited.org/resources/mail-voting/](https://statesunited.org/resources/mail-voting/) |

## 🔍 Protocol Disclosure & Non-Comparability Note
> [!NOTE]
> This benchmark evaluates autonomous web discovery and Chromium page rendering via TinyFish Search and Fetch APIs.
> It is evaluated on an open-web claim set and is **strictly separated** from ProofGraph's candidate-pool AVeriTeC benchmark table to ensure transparent and unpolluted reporting.
