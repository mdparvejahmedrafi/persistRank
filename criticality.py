"""
PersistRank — Phase 5: Service Criticality (Static Lookup)
Assigns business criticality scores (1-5) to each of the 12 Robot Shop services.
No computation needed — this is a domain-knowledge assignment based on
each service's role in the e-commerce checkout pipeline.
"""

# =====================================================================
# Step 5.1 — Service Criticality Table (Robot Shop / RS-Anomic)
# =====================================================================
#
# Robot Shop is an e-commerce microservice demo (like Sock Shop).
# Services and their roles:
#
#   web        — front-end nginx proxy, user-facing entry point
#   catalogue  — product listing service
#   cart       — shopping cart service
#   user       — user authentication / account service
#   payment    — payment processing (Paypal mock)
#   shipping   — shipping cost calculation + dispatch
#   dispatch   — order dispatch via message queue
#   ratings    — product ratings service
#   rabbitmq   — message broker between services
#   mongodb    — database for cart, user, ratings
#   mysql      — database for shipping
#   redis      — session/cache store
#
# Criticality scale (1-5):
#   5 = Direct revenue loss if down (payment failure = lost sale)
#   4 = Order pipeline broken (can't complete purchase)
#   3 = User-facing degradation (visible to customer)
#   2 = Supporting service (indirect impact)
#   1 = Non-checkout path (browsing only)

CRITICALITY = {
    "payment":    5,   # Failed payment = direct revenue loss
    "cart":       4,   # Cart failure = abandoned checkout flow
    "shipping":   3,   # Shipping calc needed before checkout confirm
    "dispatch":   3,   # Order dispatch failure = order not fulfilled
    "web":        3,   # Front-end down = entire site inaccessible
    "user":       3,   # Auth failure = can't login to purchase
    "mongodb":    3,   # Backs cart + user + ratings data
    "rabbitmq":   2,   # Async messaging, delayed impact
    "mysql":      2,   # Backs shipping service
    "redis":      2,   # Session cache, degraded performance
    "catalogue":  1,   # Product browsing only, not checkout-critical
    "ratings":    1,   # Product ratings, cosmetic feature
}

MAX_CRIT = 5

# =====================================================================
# Step 5.2 — Verify and display
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5: Service Criticality (Static Lookup)")
    print("=" * 60)

    print(f"\n  {'Service':<15s} {'Criticality':>12s} {'Normalized':>12s} {'Justification'}")
    print(f"  {'-'*75}")

    justifications = {
        "payment":   "Failed payment = direct revenue loss",
        "cart":      "Cart failure = abandoned checkout",
        "shipping":  "Needed before checkout confirmation",
        "dispatch":  "Order not fulfilled if down",
        "web":       "Front-end down = site inaccessible",
        "user":      "Auth failure = can't purchase",
        "mongodb":   "Backs cart + user + ratings",
        "rabbitmq":  "Async queue, delayed impact",
        "mysql":     "Backs shipping service only",
        "redis":     "Session cache, perf degradation",
        "catalogue": "Browsing only, not checkout path",
        "ratings":   "Cosmetic feature, no checkout impact",
    }

    for svc in sorted(CRITICALITY, key=CRITICALITY.get, reverse=True):
        c = CRITICALITY[svc]
        c_norm = c / MAX_CRIT
        print(f"  {svc:<15s} {c:>12d} {c_norm:>12.2f}   {justifications[svc]}")

    print(f"\n  Max criticality: {MAX_CRIT}")
    print(f"  Services defined: {len(CRITICALITY)}")
    print(f"\n{'=' * 60}")
    print("Phase 5 complete! (No computation needed — static lookup)")
