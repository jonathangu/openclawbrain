# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-11cd96c3-b5a0-49a5-99ba-beed78190836-window-005`
- winner mode: `graph_prior_only`
- trace hash: `sha256-e98818ee1fbfe6af19470aba80e3474e972af078ccda49d0b283bf9b3f9cdf05`
- fixture hash: `sha256-d657d23463bf41cd4159e478f5223c1f2880e97d1b0706959b1e80d3f0d4e745`
- score hash: `sha256-2f0103a5b9d888e79e6b317c22c8391790279461fd02caa2e11adb1e3cf2ef73`
- bundle hash: `sha256-f6c4513b11767fb8472a0f671dfdfbf9c0f2b6821f7f3d9097e2eaeb0c548b94`

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
- phrase hits: 0/4
- phrase hit rate: 0

| mode | turns | compile ok rate | phrase hit rate | learned route turn rate | attributed turn rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_brain | 1 | 0 | 0 | 0 | 1 |
| vector_only | 1 | 1 | 0 | 0 | 1 |
| graph_prior_only | 1 | 1 | 0 | 0 | 1 |
| learned_route | 1 | 1 | 0 | 1 | 1 |

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
| no_brain | 1 | 0 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-eaf48763997381e6d9ac148445f4fc78050abde4363c03acd4f6f65040d7cf98 |
| vector_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-85dee3c2378297a17b6cc14746c3e4a38fe66b18bcb249d40c48fea4c96971bf |
| graph_prior_only | 1 | 1 | 0/1 | 0 | 0 | 1 | 0 | 1 | sha256-984d0ed645de40fccc8e7c21d4ac53c12a7c2d0df818d5c5b7ce0af32dd54fe3 |
| learned_route | 1 | 1 | 0/1 | 1 | 0 | 1 | 0 | 2 | sha256-f347118be168f72198953c5ed967b9f01a1a55d522f5c22e24b304187d876aaf |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/1 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/1 | no | no | pack-925bd5b5 | sha256-38d584d135d3c31132e4effdf976180a19b990709d8e1e72336bccaa5f4af193 |
| graph_prior_only | turn-1 | 40 | yes | 0/1 | no | no | pack-925bd5b5 | sha256-23ea0ceda756dd0eb1a659776d530cfd033e06969c66f9e7dffa271957d095a4 |
| learned_route | turn-1 | 40 | yes | 0/1 | yes | no | pack-5dd06b32 | sha256-0d7d380b391f96ee52dcd239b11235a7f24021cf5b3b7a015dc07871dcca5d06 |
