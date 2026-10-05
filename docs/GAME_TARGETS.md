# Exact game targets and measurable success

These are the two exact-title Roblox experiences found on 5 October 2026. Game descriptions establish objectives; they are not evidence that this model can play. The action vocabularies in `configs/` are provisional until their controls and sensitivity are confirmed in the live client.

| Target | Place ID | Objective used in this repository |
| --- | --- | --- |
| [Peel a Potato — Apartment Horrors](https://www.roblox.com/games/79625784751575/Peel-a-Potato) | `79625784751575` | Finish one shipment by clearing the 2,400-potato pile and showing the completion/Caps result. |
| [Deadly Delivery — WTHHHELL BRO](https://www.roblox.com/games/125810438250765/Deadly-Delivery) | `125810438250765` | First benchmark: evacuate from floor 10 after a fresh run. This is a milestone, not a claim of total game completion. |

The [official Peel a Potato description](https://www.roblox.com/games/79625784751575/Peel-a-Potato) explains the grab → bench peel → water-channel payment loop and the 2,400-potato shipment. A useful early criterion is `potato_one_paid_cycle`, verified by a visible successful peel/deposit and payment change. That criterion must never be relabeled `potato_shipment_complete`.

The [official Deadly Delivery page with badge descriptions](https://www.roblox.com/es/games/125810438250765/Deadly-Delivery?gameSearchSessionInfo=9466c5e1-25b8-4693-8cfc-9cddb6fcfb17&isAd=false&nativeAdData=&numberOfLoadedTiles=103&page=searchPage&placeId=125810438250765&position=85&universeId=8950496606) describes evacuation badges for floors 10, 20 and 30. Freeze a specific start and evacuation goal before testing. Higher-floor achievements, cooperative survival and a complete solo run are separate tasks with separate denominators.

There is a different [Peel THE Potato](https://www.roblox.com/games/116701845804918/Peel-THE-Potato), by soviet potato republic, whose objective is finding a hidden key and escaping a vault. It is not the place targeted here. The generic obby config does not identify or certify a particular obby.

## What counts as a trial

Start each trial with the same checkpoint, declared model SHA-256, action config, window size and camera sensitivity. Record the game place, account progress, equipped upgrades, chosen route, party size, start condition and criterion before play. Progress saved from previous shipments can make later trials easier; identify it rather than treating all runs as equivalent.

A trial begins when the agent receives control in the declared start state. All subsequent deaths, timeouts, disconnects, stuck loops, capture failures and manual interventions belong in the outcome. A retry is another trial. Manual setup is permitted before the trial; any intervention after it starts makes that trial assisted. Do not discard difficult starts or only report the best clip.

For a potato paid-cycle trial, show the starting potato/payment state and the successful payment change. For a shipment trial, show shipment progress at the beginning and the game's shipment completion reward at the end. Joining a nearly completed cooperative shipment does not demonstrate an autonomous full shipment.

For a delivery evacuation trial, preserve the fresh-run start, progression and actual evacuation result. A lobby screen, loading screen, survived minute, delivery interaction, or floor counter by itself is not proof of evacuation. If another player carried the objective, report the cooperative contribution explicitly; it is not solo-agent completion.

## Long sessions and reliability

Use early runs for development. After freezing a candidate checkpoint, run a separate evaluation set that was not used to select demonstrations or tune controls. Report the exact number of trials and all outcomes, plus a 95% Wilson interval. For example, even 1/1 successes gives a lower bound of only about 20.7%; it does not establish reliable play.

A reasonable declared target is 30 fresh independent trials per criterion and at least one continuous 60-minute run with no manual recovery. These are acceptance targets, not measured results. Report valid completed cycles per minute, deaths, stuck events, interventions, longest unassisted episode, end-to-end decision latency, and peak policy-process memory. Keep incomplete runs in the denominator. Hour-long action output without visible task progress does not meet sustained-play success.

The evaluator checks complete logs, screenshot decoding, file hashes, checkpoint consistency and reviewer annotations. It cannot determine from pixels whether the reviewer was correct. `verified_wins` means *reviewed visible successes with the required files*, not a cryptographic or automated proof of gameplay. Keep the full episode evidence for independent review.
