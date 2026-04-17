# Recorded Session Replay Proof Bundle

- trace id: `live-pelican-6c8e5822-bfa9-4d00-8a6e-732666f3129e-window-173`
- winner mode: `graph_prior_only`
- trace hash: `sha256-f6e44a71ffb544349fa10e1154a65bb6e77238a611db5acd86432535b5d68dc4`
- fixture hash: `sha256-3faebbeffb8f05bd64fe046d292ad1b3475373e375c449edb9cff67872d9f497`
- score hash: `sha256-c0cb767ae01b4f3566b629474bd7aab4bd9f4e588944a57077aeaff18e1e7703`
- bundle hash: `sha256-84f78107592205b89bb11a80beaae70f83c36aec34b8b3186339fed56ee4ca3c`

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
- phrase hits: 0/8
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
| no_brain | 1 | 0 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-b418587cdea65dda940f9a601cf2fc169601499e945221393d659c55b40b8049 |
| vector_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-9cadcef19b411ea20cf7348aeb06fe310db0b329b2f50f5617fa2c298d581fe2 |
| graph_prior_only | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 1 | sha256-018733be84a185afa47d205aa5a9ed929f4b075b790e543afb45316bea300f0b |
| learned_route | 1 | 1 | 0/2 | 0 | 0 | 1 | 0 | 2 | sha256-403bdb5f645b0fca49e22459823d41ee494e7d1ec60b4088ee6c143e278e177a |

## Turn Table
| mode | turn | quality | compile ok | phrase hits | learned route | promoted | active pack | selection digest |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- |
| no_brain | turn-1 | 0 | no | 0/2 | no | no | none | none |
| vector_only | turn-1 | 40 | yes | 0/2 | no | no | pack-44dfc1ca | sha256-7c9d02fb46788a04590ce61aa833a828a0ed090caa89781b4e2e7a5504036489 |
| graph_prior_only | turn-1 | 40 | yes | 0/2 | no | no | pack-44dfc1ca | sha256-b785c450f6071e4485cd3e6b665433accc08c55564dd310de61bfab964205c4b |
| learned_route | turn-1 | 40 | yes | 0/2 | no | no | pack-68e32aa1 | sha256-41e0d163445eaff5c9003a171e10de50d30d4fc23cfd7fd74798689ffba477c0 |
