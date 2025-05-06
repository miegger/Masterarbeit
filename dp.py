import math
prices = [5,12,1,25] #--> p[0]: Preis Stab Länge 1


# max(p[j] + rec(i - j)) j element von (1, i)
# i = 4:
# max(p[1] + rec(3), p[2] + rec(2), p[3] + rec(1), p[4] + rec(0))

def best_price(prices, i, memo = None):
    if memo is None:
        memo = dict() #{1: 5, 2: 12, ....} 

    if i not in memo:
        # Basisfall der Rekursion:
        if i == 1:
            memo[i] = prices[0]
            return prices[0]

        # Rekursive Fälle
        max = 0
        for j in range(1, i + 1):
            res = prices[j - 1] + best_price(prices, i - j, memo)
            if res > max:
                max = res
        
        memo[i] = max
        return max
    else:
        return memo[i]

#print(best_price(prices, 4))




# max(p[j] + rec(i - j)) j element von (1, i)

def best_price_dp(prices, i):
    s = [None]*i

    s[0] = prices[0]

    #s = [5, None, None, None]
    #s[1] = max(prices[0] + s[0], prices[1]) 
    #s[2] = max(prices[0] + s[1], prices[1] + s[0], prices[2]) 
    #s[3] = max(prices[0] + s[2], prices[1] + s[1], prices[2] + s[0], prices[3]) 

    for len in range(1, i):        
        
        max_p = 0
        for j in range(0, len):
            res = prices[j] + s[len - j - 1] #i = 2:  prices[0] + s[1], prices[1] + s[0]
            if res > max_p:
                max_p = res
        s[len] = max(max_p, prices[len])

    return s[-1]


print(best_price_dp(prices, 4))

