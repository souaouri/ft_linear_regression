import math

mileage = []
price = []

with open("data.csv", "r") as file:
    next(file)

    for line in file:
        parts = line.strip().split(',')
        mileage.append(float(parts[0]))
        price.append(float(parts[1]))

mean_mileage = sum(mileage) / len(mileage)
mean_price = sum(price) / len(price)

std_mileage = math.sqrt(sum((x - mean_mileage) ** 2 for x in mileage) / len(mileage))
std_price = math.sqrt(sum((y - mean_price) ** 2 for y in price) / len(price))
normalized_mileage = [(m - mean_mileage) / std_mileage for m in mileage]
normalized_price = [(p - mean_price) / std_price for p in price]

def GradientDescentAlgorithm(iteration=1000, learning_rate=0.1):
    theta_0 = 0
    theta_1 = 0
    m = len(normalized_mileage)

    for i in range(iteration):
        sum_error_0 = 0
        sum_error_1 = 0
        for j in range(m):
            prediction = theta_1 * normalized_mileage[j] + theta_0
            error = prediction - normalized_price[j]

            sum_error_0 += error
            sum_error_1 += error * normalized_mileage[j]
        
        theta_0 -= learning_rate * (sum_error_0 / m)
        theta_1 -= learning_rate * (sum_error_1 / m)
    return (theta_0, theta_1)




