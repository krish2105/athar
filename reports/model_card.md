# Model card: ATHAR — triangulated marketing incrementality

> **The data behind every number in this card is simulated.** Results demonstrate the method. They are not findings about the real population.

## Model

- **Name:** ATHAR — triangulated marketing incrementality
- **Version:** 0.1.0
- **Owner:** Krishna Mathur
- **Description:** A media-mix model, a randomised-experiment analysis and an attribution model, reconciled against a known ground truth and priced as a budget decision.

## Intended use

An academic demonstration, for MAIB AI 208, that platform-reported return is not incremental return, and a measurement of how far three standard methods sit from a truth none of them can see. Read by examiners and by readers of the public portfolio.

**Out of scope:** budget decisions for any real advertiser; any claim about a named marketing channel's real effectiveness; any exchange-rate or currency claim.

## Data

- **Source:** Olist Brazilian E-Commerce (real, 96,731 orders across 85 weeks) and Criteo-UPLIFT v2.1 (real randomised trial, 13,979,592 rows), plus a simulated five-channel spend panel generated from config/dgp.yaml
- **Provenance:** simulated
- **Provenance note:** Mixed, and labelled as simulated because the strictest label wins. The Criteo result and the Olist frame are real and carry their own artifacts; everything downstream of the spend panel is simulated. athar.provenance enforces the distinction on every artifact.
- **Window:** 2017-01-02 to 2018-08-13
- **Configuration:** config/dgp.yaml, digest d9e7b488eb749462, seed 20260829

## Metrics

| Metric | Value | Split |
| --- | --- | --- |
| Intent-to-treat lift (Criteo) | 0.00115187 | full population, 13,979,592 rows, no sampling |
| Platform-reported over incremental conversions | 1.6827 | full population, 13,979,592 rows, no sampling |
| Average-ROI coverage, matched fit | 1.0 | headline panel, 85 weeks, real Olist baseline |
| Average-ROI coverage, misspecified fit | 0.6 | headline panel, 85 weeks, real Olist baseline |
| Recovery-grid fits converged | 21/40 | 40-cell factorial grid, 5 seeds per cell |
| BG/NBD maximum-likelihood fits that converged | 0/15 | full Olist base, 3 time scales x 5 penalties |
| Holdout mean absolute error, repeat purchases | 0.008527 | 16-week holdout to 2018-08-20 |
| Cost of allocating from last-click attribution | 0.3421 | headline panel, full budget, evaluated under the true response curves |

## Limitations

- No figure describes the effectiveness of any real marketing channel. The channel-spend panel, the effect sizes and the attribution bias are all constructed from a pre-registered configuration.
- The revenue baseline is treated as if it were a no-advertising counterfactual. Olist did market itself over 2017-18, so the real series already contains real media effects; layering a simulated effect on top produces a series whose simulated component has a known truth, which is the only claim made.
- The ordering of attribution bias across channels follows Blake, Nosko and Tadelis (2015), Econometrica 83(1), 155-174, who found returns to branded keyword advertising indistinguishable from zero. The magnitudes here are chosen, not measured.
- Currency is Brazilian reais, 2017-18. Every dirham figure is a stated scenario at a nominal rate and is not a measurement.
- The media-mix results are conditional on a media signal share of roughly 17% of detrended revenue variance on the headline panel. A quieter media plan is a harder recovery problem and the reported errors would grow.
- Criteo's twelve features are anonymised, so no uplift finding can be given a business interpretation, and they are heavily repeated, so the nominal row count overstates the precision available to a model fitted on them.
- Meridian was gated and runs on this machine, but no Meridian fit is reported: adapting the panel to its geo-hierarchical interface faithfully is separate work, and an unfaithful adaptation would be worse than none.
- Nothing here is legal or financial advice.
