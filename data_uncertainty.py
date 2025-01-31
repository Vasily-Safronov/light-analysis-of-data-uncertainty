import numpy as np
from scipy.optimize import minimize, differential_evolution, fmin
from scipy.stats import pearsonr


def build_parabola(x_min, x_func_max, y_min, y_func_max):
    x_min = x_min
    x_mid = x_func_max
    x_max = 2 * x_func_max - x_min

    y_min = y_min
    y_mid = y_func_max
    y_max = y_min
    
    a = (y_max - (x_max * (y_mid - y_min) + x_mid * y_min - x_min * y_mid) / (x_mid - x_min)) / (x_max * (x_max - x_min - x_mid) + x_min * x_mid)
    b = (y_mid - y_min) / (x_mid - x_min) - a * (x_min + x_mid)
    c = (x_mid * y_min - x_min * y_mid) / (x_mid - x_min) + a * x_min * x_mid
    return (a, b, c)


def parabola_by_quantile(x, y, quantile_coeffs, quantile):
    a, b, c = quantile_coeffs
    qunatile_y_pred = a * x ** 2 + b * x + c
    z = np.polyfit(x, y, 2)
    f = np.poly1d(z)
    f_mean_approx_max_x = fmin(lambda x: -f(x), 0, disp=False)[0]
    f_mean_approx_max_y = f(f_mean_approx_max_x)
    
    y_mid_right = f(x)
    if quantile < 0.5:
        y_quantile_right = f_mean_approx_max_y - (abs(qunatile_y_pred[-1] - y_mid_right[-1]))
    else:
        y_quantile_right = f_mean_approx_max_y + (abs(qunatile_y_pred[-1] - y_mid_right[-1]))
    
    a, b, c = build_parabola(x_min=x[0], x_func_max=f_mean_approx_max_x, y_min=qunatile_y_pred[0], y_func_max=y_quantile_right)
    return (a, b, c)


# Функция потерь для подбора квантиля параболы
def parabola_weight_loss(params, x, y, func_type, error_type, quantile):
    local_quantile = params[0]
    quantile_coeffs = minimize(quantile_loss, x0=[0, 0, 0],
                               args=(x, y, func_type, error_type, local_quantile),
                               options={'disp': False}).x
    
    # получаем коэффициенты конечной параболы с одинаковой вершиной
    a, b, c = parabola_by_quantile(x=x, y=y, quantile_coeffs=quantile_coeffs, quantile=quantile)
    y_pred = a * x ** 2 + b * x + c

    # считаем процент охваченный точек
    data_perc = len([y_i for y_i, y_pred_i in zip(y, y_pred) if y_i <= y_pred_i]) / len(y)
    error = abs(quantile - data_perc)
    return error


def calc_errors(error_type, y_pred, y):
    if error_type == 'least_squares':
        quantile_errors = (y - y_pred)**2
    elif error_type == 'abs':
        quantile_errors = abs(y - y_pred)
    return quantile_errors


# Функция потерь для квантильной регрессии
def quantile_loss(params, x, y, func_type, error_type, quantile):

    if func_type == 'line':
        slope, intercept = params
        y_pred = slope * x + intercept
        
    elif func_type == 'parabola':
        a, b, c = params
        y_pred = a * x**2 + b * x + c

    quantile_errors = calc_errors(error_type, y_pred, y)
    loss = np.where(y_pred >= y, (1 - quantile) * quantile_errors, quantile * quantile_errors)
    return np.sum(loss)


def line_by_quantile(x, y, quantile, quantile_coeffs, local_border, side):
    
    if quantile < 0.5:
        
        f_quantile = np.poly1d(quantile_coeffs)
        y_preds = f_quantile(local_border)
        
        z = np.polyfit(x, y, 1)
        f_mean = np.poly1d(z)
        y_mid = f_mean(local_border)
        
        left_border = y_preds[0]
        right_border = y_preds[-1]
        
        if side == 'left':
            y_left = y_mid[0] - (abs(f_quantile(x)[0] - f_mean(x)[0]))
            y_lims = [y_left, right_border]
            
        elif side == 'right':
            y_right = y_mid[-1] - (abs(f_quantile(x)[-1] - f_mean(x)[-1]))
            y_lims = [left_border, y_right]
            
        a, b = np.polyfit(np.array([local_border[0], local_border[-1]]), np.array([y_lims[0], y_lims[-1]]), 1)
        
    elif quantile >= 0.5:
        f_quantile = np.poly1d(quantile_coeffs)
        y_preds = f_quantile(local_border)
        
        z = np.polyfit(x, y, 1)
        f_mean = np.poly1d(z)
        y_mid = f_mean(local_border)
        
        left_border = y_preds[0]
        right_border = y_preds[-1]
        
        if side == 'left':
            y_left = y_mid[0] + (abs(f_quantile(x)[0] - f_mean(x)[0]))
            y_lims = [y_left, right_border]
            
        elif side == 'right':
            y_right = y_mid[-1] + (abs(f_quantile(x)[-1] - f_mean(x)[-1]))
            y_lims = [left_border, y_right]

        a, b = np.polyfit(np.array([local_border[0], local_border[-1]]), np.array([y_lims[0], y_lims[-1]]), 1)

    return a, b


# Функция потерь для подбора квантиля прямой
def line_weight_loss(params, x, y, func_type, error_type, quantile, local_border, side):
    local_quantile = params[0]
    quantile_coeffs = minimize(quantile_loss, x0=[0, 0],
                               args=(x, y, func_type, error_type, local_quantile),
                               options={'disp': False}).x
    
    # получаем коэффициенты конечной параболы с одинаковой вершиной
    a, b = line_by_quantile(x=x, y=y, quantile=quantile,
                            quantile_coeffs=quantile_coeffs,
                            local_border=local_border,
                            side=side)

    y_pred = a * x + b
    
    # считаем процент охваченный точек
    data_perc = len([y_i for y_i, y_pred_i in zip(y, y_pred) if y_i <= y_pred_i]) / len(y)
    error = abs(quantile - data_perc)
    return error


class Quantile_regression:
    
    calc_errors = calc_errors
    quantile_loss = quantile_loss
    parabola_weight_loss = parabola_weight_loss
    line_weight_loss = line_weight_loss
    
    def __init__(self, x, y, func_type='line', error_type='abs', quantiles=[0.5, 0.95], correction=False, local_border=None):
        self.func_type = func_type
        self.coeffs = {}
        self.preds = {}
        self.r2 = None
        self.run_quantile_reg(x, y, func_type, error_type, quantiles, correction, local_border)
    
    def predict(self, x, quantile):
        
        x = np.array(x)
        
        if self.func_type == 'line':
            slope, intercept = self.coeffs[quantile]
            y_pred = slope * x + intercept

        elif self.func_type == 'parabola':
            a, b, c = self.coeffs[quantile]
            y_pred = a * x**2 + b * x + c

        return y_pred

    # апроксимация функцией по квантилям
    def run_quantile_reg(self, x, y, func_type, error_type, quantiles, correction, local_border):
        
        x = np.array(x)
        
        if func_type == 'line':
            if correction == True:
                local_coeffs_dct = {}
                preds = {}
                
                for quantile in quantiles:
                    initial_params = [0, 0]
                    slope, intercept = minimize(quantile_loss, x0=initial_params,
                                                args=(x, y, func_type, error_type, quantile),
                                                options={'disp': False}).x
                    local_coeffs_dct[quantile] = [slope, intercept]
                    preds[quantile] = slope * np.array(local_border) + intercept
                
                left = preds[quantiles[0]][0] > preds[quantiles[1]][0]
                right = preds[quantiles[0]][-1] > preds[quantiles[1]][-1]
                
                if left or right:

                    if left:
                        side = 'left'

                    if right:
                        side = 'right'
                        
                    for quantile in quantiles:

                        if quantile >= 0.5:
                            bounds = [(0.5, 1)]
                            quantile_param = [0.6]
                        elif quantile < 0.5:
                            bounds = [(0, 0.5)]
                            quantile_param = [0.1]

                        opt_quantile = differential_evolution(line_weight_loss, x0=quantile_param,
                                                                bounds=bounds, atol=0.0001,
                                                                args=(x, y, func_type, 'least_squares',
                                                                    quantile, local_border, side),
                                                                disp=False).x[0]
                        
                        initial_params = [0, 0]
                        slope, intercept = minimize(quantile_loss, x0=initial_params,
                                                    args=(x, y, func_type, 'least_squares', opt_quantile),
                                                    options={'disp': False}).x
                        
                        slope, intercept = line_by_quantile(x=x, y=y, quantile=opt_quantile,
                                                quantile_coeffs=[slope, intercept],
                                                local_border=local_border,
                                                side=side)
                        
                        self.preds[quantile] = slope * x + intercept
                        self.coeffs[quantile] = [slope, intercept]
                else:
                    self.preds = preds
                    self.coeffs = local_coeffs_dct

            else:
                error_type = 'least_squares'
                for quantile in quantiles:
                    initial_params = [0, 0]
                    slope, intercept = minimize(quantile_loss, x0=initial_params,
                                                args=(x, y, func_type, error_type, quantile),
                                                options={'disp': False}).x
                    self.preds[quantile] = slope * x + intercept
                    self.coeffs[quantile] = [slope, intercept]

            z = np.polyfit(x, y, 1)
            f = np.poly1d(z)
            self.preds['mean'] = f(x)
            self.coeffs['mean'] = z
            self.r2 = pearsonr(x, y)[0] ** 2
            
        elif func_type == 'parabola':
            if correction == True:
                error_type = 'least_squares'
                for quantile in quantiles:
                    
                    if quantile >= 0.5:
                        bounds = [(0.5, 1)]
                        quantile_param = [0.6]
                        
                    elif quantile < 0.5:
                        bounds = [(0, 0.5)]
                        quantile_param = [0.1]
                        
                    opt_quantile = differential_evolution(parabola_weight_loss, x0=quantile_param,
                                        bounds=bounds, atol=0.0001,
                                        args=(x, y, func_type, error_type, quantile), disp=False).x[0]
                    # построение квантильной регрессии
                    initial_params = [0, 0, 0]
                    quantile_coeffs = minimize(quantile_loss, x0=initial_params,
                                            args=(x, y, func_type, error_type, opt_quantile),
                                            options={'disp': False}).x
                    a, b, c = parabola_by_quantile(x, y, quantile_coeffs, quantile)
                    self.preds[quantile] = a * x ** 2 + b * x + c
                    self.coeffs[quantile] = [a, b, c]
                    
            else:
                error_type = 'least_squares'
                for quantile in quantiles:
                    # построение квантильной регрессии
                    initial_params = [0, 0, 0]
                    a, b, c = minimize(quantile_loss, x0=initial_params,
                                            args=(x, y, func_type, error_type, quantile),
                                            options={'disp': False}).x
                    self.preds[quantile] = a * x ** 2 + b * x + c
                    self.coeffs[quantile] = [a, b, c]
            
            z = np.polyfit(x, y, 2)
            f = np.poly1d(z)
            self.preds['mean'] = f(x)
            self.coeffs['mean'] = z
            self.r2 = pearsonr(y, f(x))[0] ** 2
            
        return self.preds, self.coeffs


class Classic_regression:
    def __init__(self, x, y, func_type, n_sigm):
        self.coeffs = np.array([])
        self.pred_interval = None
        self.r2 = None
        self.fit(x, y, func_type, n_sigm)
        
    def fit(self, x, y, func_type, n_sigm=1.96):
        x = np.array(x)
        
        if func_type == 'line':
            coeffs = np.polyfit(x, y, 1)
            f = np.poly1d(coeffs)
            y_pred = f(x)
            r2 = pearsonr(x, y)[0] ** 2
            
        elif func_type == 'parabola':
            coeffs = np.polyfit(x, y, 2)
            f = np.poly1d(coeffs)
            y_pred = f(x)
            r2 = pearsonr(y, y_pred)[0] ** 2
            
        # estimate stdev of yhat
        sum_errs = np.sum((y - y_pred)**2)
        stdev = np.sqrt(1/(len(y)-2) * sum_errs)
        # calculate prediction interval
        interval = n_sigm * stdev
        
        # sf = stdev * np.sqrt(1 + 1/len(x) + ((x - np.mean(x))**2 / np.sum((x - np.mean(x))**2)))
        # interval = n_sigm * sf

        self.coeffs = coeffs
        self.pred_interval = interval
        self.r2 = r2
 
    def predict(self, x):
        x = np.array(x)
        f = np.poly1d(self.coeffs)
        return f(x)