import math as maths
import matplotlib.pyplot as plt

# Global Variables
TIME_LIMIT = 3_600 * 24
CLICK_SPEED = 5

class IdleSim:
    def __init__(self, eps=0.15):
        self.cookies = 0
        self.ltcookies = 0
        self.cps = 0
        # eps is the multiplier in cost for each new building (cookie clicker: eps=0.15)
        self.eps = eps
        self.buildings = {"clicker": 0, "build1": 0, "build2": 0, "build3": 0, "build4": 0}
        self.building_costs = {"clicker": 10, "build1": 100, "build2": 1_000, "build3": 10_000, "build4": 100_000}
        self.building_cps = {"clicker": 1, "build1": 10, "build2": 100, "build3": 1_000, "build4": 10_000}

    def clickCookie(self, clicks: int):
        self.cookies += clicks
        self.ltcookies += clicks

    def buyBuilding(self, build: str):
        if self.cookies >= self.building_costs[build]:
            self.cookies -= maths.floor(self.building_costs[build])
            self.buildings[build] += 1
            self.building_costs[build] = self.building_costs[build] * (1 + self.eps)
            # Update cps
            self.cps = sum([self.buildings[b] * self.building_cps[b] for b in self.buildings])

    def updateCookies(self):
        self.cookies += self.cps
        self.ltcookies += self.cps


# Test Strategies
t = 0
# One: Only click
cc1 = IdleSim()
# Two: Buy the cheapest building available otherwise click cookie
cc2 = IdleSim()
# Three: If there are enough cookies the strat buys the building with the smallest cost / cps ratio
cc3 = IdleSim()
# Four: Spam Clickers
cc4 = IdleSim()
# Five: Spam most expensive building
cc5 = IdleSim()
cookies_hist = {f"strat{i}": [] for i in range(1, 6)}
lt_cookie_hist = {f"strat{i}": [] for i in range(1, 6)}
cps_hist = {f"strat{i}": [] for i in range(1, 6)}
while t < TIME_LIMIT:
    # Strat 1
    cc1.clickCookie(CLICK_SPEED)

    # Strat 2
    # Checks if anything can be bought
    m = min([cc2.building_costs[b] for b in cc2.building_costs])
    if cc2.cookies >= m:
        while cc2.cookies >= m:
            for b in cc2.building_costs:
                if cc2.building_costs[b] == m:
                    cc2.buyBuilding(b)
                    break
            m = min([cc2.building_costs[b] for b in cc2.building_costs])
    else:
        cc2.clickCookie(CLICK_SPEED)

    # Strat 3
    # Find desired building
    min_ratio = cc3.building_costs["clicker"] / cc3.building_cps["clicker"]
    min_build = "clicker"
    min_cost = cc3.building_costs["clicker"]
    for b in cc3.buildings:
        if cc3.building_costs[b] / cc3.building_cps[b] < min_ratio:
            min_ratio = cc3.building_costs[b] / cc3.building_cps[b]
            min_build = b
            min_cost = cc3.building_costs[b]
    # Check if it can buy the desired building
    if cc3.cookies >= min_cost:
        while cc3.cookies >= min_cost:
            cc3.buyBuilding(min_build)
            min_ratio = cc3.building_costs["clicker"] / cc3.building_cps["clicker"]
            min_build = "clicker"
            min_cost = cc3.building_costs["clicker"]
            for b in cc3.buildings:
                if cc3.building_costs[b] / cc3.building_cps[b] < min_ratio:
                    min_ratio = cc3.building_costs[b] / cc3.building_cps[b]
                    min_build = b
                    min_cost = cc3.building_costs[b]
    else:
        cc3.clickCookie(CLICK_SPEED)

    # Strat 4
    if cc4.cookies >= cc4.building_costs["clicker"]:
        while cc4.cookies >= cc4.building_costs["clicker"]:
            cc4.buyBuilding("clicker")
    else:
        cc4.clickCookie(CLICK_SPEED)

    # Strat 5
    if cc5.cookies >= cc5.building_costs["build4"]:
        while cc5.cookies >= cc5.building_costs["build4"]:
            cc5.buyBuilding("build4")
    else:
        cc5.clickCookie(CLICK_SPEED)

    # Update cookies for each strat
    cc1.updateCookies()
    cookies_hist["strat1"].append(maths.log(1 + cc1.cookies, 10))
    cps_hist["strat1"].append(maths.log(1 + cc1.cps, 10))
    lt_cookie_hist["strat1"].append(maths.log(1 + cc1.ltcookies, 10))
    cc2.updateCookies()
    cookies_hist["strat2"].append(maths.log(1 + cc2.cookies, 10))
    cps_hist["strat2"].append(maths.log(1 + cc2.cps, 10))
    lt_cookie_hist["strat2"].append(maths.log(1 + cc2.ltcookies, 10))
    cc3.updateCookies()
    cookies_hist["strat3"].append(maths.log(1 + cc3.cookies, 10))
    cps_hist["strat3"].append(maths.log(1 + cc3.cps, 10))
    lt_cookie_hist["strat3"].append(maths.log(1 + cc3.ltcookies, 10))
    cc4.updateCookies()
    cookies_hist["strat4"].append(maths.log(1 + cc4.cookies, 10))
    cps_hist["strat4"].append(maths.log(1 + cc4.cps, 10))
    lt_cookie_hist["strat4"].append(maths.log(1 + cc4.ltcookies, 10))
    cc5.updateCookies()
    cookies_hist["strat5"].append(maths.log(1 + cc5.cookies, 10))
    cps_hist["strat5"].append(maths.log(1 + cc5.cps, 10))
    lt_cookie_hist["strat5"].append(maths.log(1 + cc5.ltcookies, 10))
    # Update time
    t += 1

if __name__ == "__main__":
    for i in range(1, 6):
        print(f"Strat {i} | {round(cps_hist[f'strat{i}'][-1], 3)} | {round(lt_cookie_hist[f'strat{i}'][-1], 3)}")


    # Plot the different strategies
    colours = ["red", "orange", "yellow", "green", "blue", "purple"]
    plt.style.use("ggplot")
    for i in range(2, 6):
        plt.plot([t for t in range(1, TIME_LIMIT + 1)], cps_hist[f"strat{i}"], color=colours[(i-1) % 6])

    plt.xlabel("In Game Time")
    plt.ylabel("log_10(1+cps)")
    plt.ylim((5.8, 6))
    plt.title("Strat Comparison")
    plt.legend([s for s in cookies_hist][1:])
    plt.show()