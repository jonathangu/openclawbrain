# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-4654a9d2-02db-4eaa-a316-86d131e91df8-window-002`
- winner mode: `graph_prior_only`
- trace hash: `sha256-0107560e9fd434b7938c996a94e09516e9330df1381928365035d337054775c9`
- fixture hash: `sha256-ccf24038ed94c209310a49ea52fc2105449214d461e66d2dc1493bec54050346`
- score hash: `sha256-464f7bf231b3481c2077829a89a70cb1a93063d1b99a3487706f21a87960f1c1`
- bundle hash: `sha256-6d08ebc290cf7724bc6a70c506c75d7bab6997e483495a632c152124e0ca4361`

## Ranking
| rank | mode | quality score |
| --- | --- | ---: |
| 1 | graph_prior_only | 40 |
| 2 | learned_route | 40 |
| 3 | vector_only | 40 |
| 4 | no_brain | 0 |

## Coverage Snapshot
- compile ok turns: 3/4
- compile ok rate: 0.75
- phrase hits: 0/12
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 0 | 1 |

## Hardening Snapshot
- compile failures: 1/4
- compile failure rate: 0.25
- warnings: 5
- promotions: 0

| mode | warnings | compile failures | promotions | export turns | attributed turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 1 | 0 | 1 | 1 |
| vector_only | 1 | 0 | 0 | 1 | 1 |
| graph_prior_only | 1 | 0 | 0 | 1 | 1 |
| learned_route | 2 | 0 | 0 | 1 | 1 |

## Mode Table
| mode | turns | compile ok | phrase hits | learned route turns | promotions | export turns | human labels | warnings | score hash |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| no_brain | 1 | 0 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-33cd39ad77bdbc78ea0e62a163e0f69b70fc53f35c07ff18076ffb99dd86c22c |
| vector_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-f7e98130fe1f7b105f69289588a1b1912b639e23e2f5595222beecfd66a3e8b3 |
| graph_prior_only | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 1 | sha256-7561e926d39de2460b81b76e75c538b5d38ed4e6ce88e0f0b252ea1f9ef4cccb |
| learned_route | 1 | 1 | 0/3 | 0 | 0 | 1 | 0 | 2 | sha256-5415496fb27a0fa2b354432d2f607efadeb5a8d1354d977fbd027ef0d738045d |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/3 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fa1707ea | sha256-2fcea725982b356c7df24d5f6a864385256cdf7fa5122d91ce05db3f171d5cb1 |
| graph_prior_only | turn-1 | 40 | yes | 0/3 | no | no | pack-fa1707ea | sha256-13a5cf272db4206cf64e09f1da971ed7472f111a2116cf0e0f67bd38687130f7 |
| learned_route | turn-1 | 40 | yes | 0/3 | no | no | pack-fa1707ea | sha256-2fcea725982b356c7df24d5f6a864385256cdf7fa5122d91ce05db3f171d5cb1 |
