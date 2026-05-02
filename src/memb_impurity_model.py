"""
Модель переноса примесей и воды через мембрану PEMFC.

Модуль содержит ту же расчётную логику, что и демонстрационный notebook:
- параметры БТЭ и газовых состояний;
- расчёт crossover O2, N2, H2;
- расчёт переноса воды через мембрану;
- вывод потоков в формате, близком к Amesim;
- оценку накопления примесей при закрытой рециркуляции анода.

Назначение: учебно-исследовательский пример для инженерного портфолио.
"""

# -----------------------------
# ПАРАМЕТРЫ БТЭ
# -----------------------------

# Количество ячеек БТЭ, шт.
Ncell = 250

# Активная площадь одной ячейки БТЭ, см^2.
Scell = 417.74

# Толщина мембраны, мм.
lmemb = 0.125

# Эквивалентная масса иономера, г/моль.
# По смыслу: масса сухого полимера на 1 моль сульфогрупп SO3-.
EW = 1100.0

# Плотность материала мембраны, кг/м^3.
rho_memb = 2000.0

# -----------------------------
# ВХОДНЫЕ СОСТОЯНИЯ AMESIM
# -----------------------------

# Анод: температура смеси, К.
ANODE_T_K = 301.582

# Анод: абсолютное давление смеси, Па.
ANODE_P_PA = 199404.0

# Анод: молярные доли компонентов смеси.
ANODE_X_O2 = 0.00
ANODE_X_N2 = 0.0235389
ANODE_X_H2O = 0.0188723
ANODE_X_H2 = 1 - ANODE_X_O2 - ANODE_X_N2 - ANODE_X_H2O

# Катод: температура смеси, К.
CATHODE_T_K = 353.0

# Катод: абсолютное давление смеси, Па.
CATHODE_P_PA = ANODE_P_PA + 10e3

# Катод: молярные доли компонентов смеси.
# Здесь по умолчанию задан демонстрационный состав катодной смеси. Для примера влажного воздуха можно использовать значения из комментариев справа.
CATHODE_X_O2 = 0.0 # 0.1250
CATHODE_X_N2 = 0.7 # 0.4700
CATHODE_X_H2O = 0.3 # 0.4050
CATHODE_X_H2 = 1 - CATHODE_X_O2 - CATHODE_X_N2 - CATHODE_X_H2O  # 0.0

# -----------------------------
# РАСЧЁТНЫЙ ТОК
# -----------------------------

# Ток БТЭ для демонстрационного расчёта, А.
CURRENT_A = 300.0

# -----------------------------
# ПАРАМЕТРЫ ЗАКРЫТОЙ РЕЦИРКУЛЯЦИИ
# -----------------------------

# Стехиометрическое число на входе в БТЭ.
# По модели: dm1 = lambda * dm_H2.
ANODE_STOICH_LAMBDA = 1.5

# Коэффициент после влагоотделителя, %.
# Используется формула humid_out = (eff/100) * humid_in.
# 100% означает ту же относительную влажность, 0% означает полное осушение.
WATER_SEPARATOR_EFF_PERCENT = 93.0

# Горизонт накопления примесей, с.
ACCUMULATION_TIME_S = 100.0

# -----------------------------
# ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ
# -----------------------------

# Постоянная Фарадея, Кл/моль.
FARADAY = 96485.3415

# Универсальная газовая постоянная, Дж/(моль·К).
R_UNIV = 8.3144

# Молярные массы компонентов, г/моль.
M_H2_G_MOL = 2.016
M_N2_G_MOL = 28.01
M_H2O_G_MOL = 18.015
M_O2_G_MOL = 31.998


from dataclasses import dataclass, asdict
from math import exp, log10
from typing import Dict

import pandas as pd


@dataclass
class BTEParams:
    # Количество ячеек БТЭ, шт.
    ncell: int = Ncell

    # Площадь одной ячейки БТЭ, см^2.
    scell_cm2: float = Scell

    # Толщина мембраны, мм.
    lmemb_mm: float = lmemb

    # Эквивалентная масса иономера, г/моль.
    ew_g_per_mol: float = EW

    # Плотность мембраны, кг/м^3.
    rho_memb_kg_m3: float = rho_memb

    @property
    def area_m2_total(self) -> float:
        # Суммарная активная площадь БТЭ, м^2.
        return (self.scell_cm2 / 10000.0) * self.ncell

    @property
    def thickness_m(self) -> float:
        # Толщина мембраны, м.
        return self.lmemb_mm / 1000.0

    @property
    def c_so3_mol_m3(self) -> float:
        # Концентрация неподвижных сульфогрупп SO3- в мембране, моль/м^3.
        return self.rho_memb_kg_m3 / (self.ew_g_per_mol / 1000.0)


@dataclass
class GasState:
    # Температура газовой смеси, К.
    temperature_K: float

    # Полное давление смеси, Па.
    pressure_Pa: float

    # Молярные доли компонентов.
    x_h2: float
    x_n2: float
    x_h2o: float
    x_o2: float

    @property
    def p_h2(self) -> float:
        # Парциальное давление водорода, Па.
        return self.pressure_Pa * self.x_h2

    @property
    def p_n2(self) -> float:
        # Парциальное давление азота, Па.
        return self.pressure_Pa * self.x_n2

    @property
    def p_h2o(self) -> float:
        # Парциальное давление водяного пара, Па.
        return self.pressure_Pa * self.x_h2o

    @property
    def p_o2(self) -> float:
        # Парциальное давление кислорода, Па.
        return self.pressure_Pa * self.x_o2

    def validate(self) -> None:
        # Проверка, что сумма молярных долей близка к 1.
        total = self.x_h2 + self.x_n2 + self.x_h2o + self.x_o2
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Сумма молярных долей должна быть 1.0, сейчас {total}")


@dataclass
class SC3Result:
    # Газовый crossover через мембрану, моль/с.
    n_o2_to_anode_mol_s: float
    n_n2_to_anode_mol_s: float
    n_h2_to_cathode_mol_s: float

    # Потоки воды через мембрану, моль/с.
    n_h2o_diff_to_anode_mol_s: float
    n_h2o_drag_to_cathode_mol_s: float
    n_h2o_net_to_anode_mol_s: float

    # Реакционный расход водорода на аноде, моль/с.
    n_h2_consumption_mol_s: float

    # Внутренние расчётные величины.
    lambda_membrane_anode: float
    lambda_membrane_cathode: float
    phi_w_anode: float
    phi_w_cathode: float
    phi_w_avg: float
    a_h2o_anode: float
    a_h2o_cathode: float
    p_sat_anode_Pa: float
    p_sat_cathode_Pa: float
    n_drag: float


def result_to_dataframe(result: SC3Result) -> pd.DataFrame:
    # Перевод результата в таблицу для удобного просмотра в ноутбуке.
    return pd.DataFrame(
        {"variable": list(asdict(result).keys()), "value": list(asdict(result).values())}
    )


def water_saturation_pressure_Pa(T_K: float) -> float:
    # Давление насыщенного водяного пара, Па.
    # Формула воспроизводит вид зависимости из модели SC_3.
    return 133.32 * 10 ** (
        29.8605
        - 3.1522e3 / T_K
        - 7.3037 * log10(T_K)
        + 2.4247e-9 * T_K
        + 1.809e-6 * T_K * T_K
    )


def membrane_water_activity(p_h2o_Pa: float, T_K: float) -> float:
    # Активность воды как отношение парциального давления пара к давлению насыщения.
    return p_h2o_Pa / water_saturation_pressure_Pa(T_K)


def membrane_water_content(activity: float) -> float:
    # Водосодержание мембраны lambda_mem.
    # Piecewise-зависимость воспроизводит форму из SC_3.
    a = activity
    if a <= 1.0:
        return 0.043 + 17.81 * a - 39.85 * a**2 + 36.0 * a**3
    if a <= 3.0:
        return 14.0 + 1.4 * (a - 1.0)
    return 16.8


def membrane_water_volume_fraction(
    lambda_mem: float,
    ew_g_per_mol: float = EW,
    rho_memb_kg_m3: float = rho_memb,
) -> float:
    # Объёмная доля воды в мембране.
    # 1.8e-5 м^3/моль — молярный объём воды.
    # V_SO3 вычисляется из эквивалентной массы и плотности сухого иономера.
    v_w = 1.8e-5
    v_so3 = (ew_g_per_mol / 1000.0) / rho_memb_kg_m3
    return lambda_mem * v_w / (v_so3 + lambda_mem * v_w)


def electro_osmotic_drag_coefficient(lambda_avg: float) -> float:
    # Коэффициент electro-osmotic drag.
    return 2.5 * lambda_avg / 22.0


def water_diffusivity_membrane_m2_s(lambda_avg: float, T_K: float) -> float:
    # Эффективный коэффициент диффузии воды в мембране, м^2/с.
    # Восстановлен как piecewise-функция по lambda_avg с температурным множителем из SC_3.
    lmb = lambda_avg
    if lmb < 2.0:
        base = lmb * 1.0e-9
    elif lmb <= 3.0:
        base = 1.0e-9 * (1.0 + 2.0 * (lmb - 2.0))
    elif lmb <= 4.5:
        base = 1.0e-9 * (3.0 - 1.67 * (lmb - 3.0))
    else:
        base = 1.25e-9
    return base * exp(2416.0 * (1.0 / 298.15 - 1.0 / T_K))



def sc3_membrane_model(
    current_A: float,
    cathode: GasState,
    anode: GasState,
    params: BTEParams | None = None,
) -> SC3Result:
    # Основной расчёт мембранных потоков в SC_3.
    params = params or BTEParams()

    # Проверка корректности состава на обеих сторонах.
    cathode.validate()
    anode.validate()

    # Давление насыщенного пара воды на катоде и аноде, Па.
    p_sat_c = water_saturation_pressure_Pa(cathode.temperature_K)
    p_sat_a = water_saturation_pressure_Pa(anode.temperature_K)

    # Активность воды на катодной и анодной сторонах.
    activity_c = membrane_water_activity(cathode.p_h2o, cathode.temperature_K)
    activity_a = membrane_water_activity(anode.p_h2o, anode.temperature_K)

    # Локальное водосодержание мембраны на двух сторонах.
    lambda_c = membrane_water_content(activity_c)
    lambda_a = membrane_water_content(activity_a)
    lambda_avg = 0.5 * (lambda_c + lambda_a)

    # Объёмная доля воды в мембране.
    phi_c = membrane_water_volume_fraction(lambda_c, params.ew_g_per_mol, params.rho_memb_kg_m3)
    phi_a = membrane_water_volume_fraction(lambda_a, params.ew_g_per_mol, params.rho_memb_kg_m3)
    phi_avg = membrane_water_volume_fraction(lambda_avg, params.ew_g_per_mol, params.rho_memb_kg_m3)

    # Геометрия мембраны.
    area = params.area_m2_total
    lm = params.thickness_m

    # Разности парциальных давлений для crossover, Па.
    # O2 и N2 идут из катода в анод.
    # H2 идёт из анода в катод.
    dp_o2 = cathode.p_o2 - anode.p_o2
    dp_n2 = cathode.p_n2 - anode.p_n2
    dp_h2 = anode.p_h2 - cathode.p_h2

    # Газовые пермеансы мембраны.
    # Коэффициенты и температурные экспоненты заданы в инженерной форме и сопоставлены с Amesim.
    # Для газового crossover используется средняя объёмная доля воды phi_w_avg.
    k_o2 = (
        (0.11 + 1.9 * phi_avg)
        * 1e-14
        * exp(22000.0 / R_UNIV * (1.0 / 303.0 - 1.0 / cathode.temperature_K))
        * area
        / lm
    )
    k_n2 = (
        (0.0295 + 1.21 * phi_avg - 1.93 * phi_avg**2)
        * 1e-14
        * exp(24000.0 / R_UNIV * (1.0 / 303.0 - 1.0 / cathode.temperature_K))
        * area
        / lm
    )
    k_h2 = (
        (0.29 + 2.2 * phi_avg)
        * 1e-14
        * exp(21000.0 / R_UNIV * (1.0 / 303.0 - 1.0 / cathode.temperature_K))
        * area
        / lm
    )

    # Потоки crossover через мембрану, моль/с.
    n_o2 = k_o2 * dp_o2
    n_n2 = k_n2 * dp_n2
    n_h2 = k_h2 * dp_h2

    # Коэффициент electro-osmotic drag.
    n_drag = electro_osmotic_drag_coefficient(lambda_avg)

    # Диффузия воды через мембрану.
    # Для температурного множителя используется температура катодной стороны,
    # как в реконструкции исходной логики SC_3.
    d_w = water_diffusivity_membrane_m2_s(lambda_avg, cathode.temperature_K)
    n_h2o_diff = d_w * params.c_so3_mol_m3 * ((lambda_c - lambda_a) / lm) * area

    # Electro-osmotic drag тянет воду с анода к катоду.
    n_h2o_drag_to_cathode = n_drag * current_A * params.ncell / FARADAY

    # Суммарный поток воды в сторону анода.
    n_h2o_net_to_anode = n_h2o_diff - n_h2o_drag_to_cathode

    # Реакционный расход H2 на аноде.
    n_h2_consumption = -(current_A * params.ncell) / (2.0 * FARADAY)

    return SC3Result(
        n_o2_to_anode_mol_s=n_o2,
        n_n2_to_anode_mol_s=n_n2,
        n_h2_to_cathode_mol_s=n_h2,
        n_h2o_diff_to_anode_mol_s=n_h2o_diff,
        n_h2o_drag_to_cathode_mol_s=n_h2o_drag_to_cathode,
        n_h2o_net_to_anode_mol_s=n_h2o_net_to_anode,
        n_h2_consumption_mol_s=n_h2_consumption,
        lambda_membrane_anode=lambda_a,
        lambda_membrane_cathode=lambda_c,
        phi_w_anode=phi_a,
        phi_w_cathode=phi_c,
        phi_w_avg=phi_avg,
        a_h2o_anode=activity_a,
        a_h2o_cathode=activity_c,
        p_sat_anode_Pa=p_sat_a,
        p_sat_cathode_Pa=p_sat_c,
        n_drag=n_drag,
    )



COMPONENT_MOLAR_MASSES_G_MOL = {
    "o2": M_O2_G_MOL,
    "n2": M_N2_G_MOL,
    "h2o": M_H2O_G_MOL,
    "h2": M_H2_G_MOL,
}


def sc3_amesim_flux_dataframe(result: SC3Result) -> pd.DataFrame:
    # Потоки компонентов в формате, близком к панели Amesim.
    h2_source_to_anode = result.n_h2_consumption_mol_s - result.n_h2_to_cathode_mol_s

    rows = [
        ("effective molar flux (1)", result.n_o2_to_anode_mol_s, "mol/s", "O2 в анод"),
        ("effective molar flux (2)", result.n_n2_to_anode_mol_s, "mol/s", "N2 в анод"),
        ("effective molar flux (3)", result.n_h2o_net_to_anode_mol_s, "mol/s", "H2O в анод"),
        ("effective molar flux (4)", h2_source_to_anode, "mol/s", "H2 на аноде: реакция + crossover"),
    ]

    mass_flow_port3_g_s = (
        result.n_o2_to_anode_mol_s * COMPONENT_MOLAR_MASSES_G_MOL["o2"]
        + result.n_n2_to_anode_mol_s * COMPONENT_MOLAR_MASSES_G_MOL["n2"]
        + result.n_h2o_net_to_anode_mol_s * COMPONENT_MOLAR_MASSES_G_MOL["h2o"]
        + h2_source_to_anode * COMPONENT_MOLAR_MASSES_G_MOL["h2"]
    )
    rows.append(("mass flow rate at port 3", mass_flow_port3_g_s, "g/s", "Суммарный массовый поток источника"))

    return pd.DataFrame(rows, columns=["title", "value", "unit", "meaning"])


def relative_humidity_from_molar_fraction(
    x_h2o: float,
    pressure_Pa: float,
    temperature_K: float,
) -> tuple[float, float]:
    # Относительная влажность и давление насыщенного пара.
    p_sat = water_saturation_pressure_Pa(temperature_K)
    humidity = x_h2o * pressure_Pa / p_sat
    return humidity, p_sat


def closed_recirc_impurity_accumulation_dataframe(
    result: SC3Result,
    anode: GasState,
    lambda_stoich: float,
    water_separator_eff_percent: float,
    accumulation_time_s: float,
) -> pd.DataFrame:
    # Баланс накопления примесей при закрытой рециркуляции.
    eff_fraction = max(0.0, water_separator_eff_percent / 100.0)

    # Реакционный расход водорода в положительном виде.
    n_h2_consumption_abs = -result.n_h2_consumption_mol_s

    # Полное восполнение расхода H2 чистым водородом.
    n_h2_makeup_pure_mol_s = n_h2_consumption_abs

    # Требуемый расход на входе в БТЭ по заданному lambda.
    n_h2_inlet_to_bte_mol_s = lambda_stoich * n_h2_consumption_abs

    # Требуемая доля рециркуляции для достижения заданного входного расхода.
    n_h2_required_from_recirc_mol_s = max(
        0.0, n_h2_inlet_to_bte_mol_s - n_h2_makeup_pure_mol_s
    )

    # Относительная влажность на аноде до влагоотделителя.
    humidity_in, p_sat_anode = relative_humidity_from_molar_fraction(
        x_h2o=anode.x_h2o,
        pressure_Pa=anode.pressure_Pa,
        temperature_K=anode.temperature_K,
    )

    # Формула из постановки пользователя.
    humidity_out_raw = eff_fraction * humidity_in

    # Газовая фаза после влагоотделителя ограничивается насыщением.
    humidity_out = min(1.0, max(0.0, humidity_out_raw))
    x_h2o_after_separator = min(
        0.999999, humidity_out * p_sat_anode / anode.pressure_Pa
    )

    # Доля воды, остающаяся в контуре после влагоотделителя.
    water_remaining_fraction = 0.0
    if abs(humidity_in) > 1e-16:
        water_remaining_fraction = humidity_out / humidity_in

    # Потоки примесей, остающиеся в контуре после влагоотделителя.
    n_o2_accumulation_mol_s = result.n_o2_to_anode_mol_s
    n_n2_accumulation_mol_s = result.n_n2_to_anode_mol_s
    n_h2o_after_separator_mol_s = (
        result.n_h2o_net_to_anode_mol_s * water_remaining_fraction
    )
    n_h2o_removed_mol_s = (
        result.n_h2o_net_to_anode_mol_s - n_h2o_after_separator_mol_s
    )

    n_total_impurity_accumulation_mol_s = (
        n_o2_accumulation_mol_s
        + n_n2_accumulation_mol_s
        + n_h2o_after_separator_mol_s
    )

    rows = [
        ("n_h2_consumption_abs_mol_s", n_h2_consumption_abs, "mol/s", "Положительный расход H2 на реакцию в БТЭ"),
        ("n_h2_makeup_pure_mol_s", n_h2_makeup_pure_mol_s, "mol/s", "Чистый H2, полностью компенсирующий реакционный расход"),
        ("n_h2_inlet_to_bte_mol_s", n_h2_inlet_to_bte_mol_s, "mol/s", "Требуемый расход на входе в БТЭ: dm1 = lambda * dm_H2"),
        ("n_h2_required_from_recirc_mol_s", n_h2_required_from_recirc_mol_s, "mol/s", "Требуемая доля H2 из рециркуляции"),
        ("humidity_in", humidity_in, "-", "Относительная влажность на аноде до влагоотделителя"),
        ("humidity_out_raw", humidity_out_raw, "-", "Влажность по формуле humid_out = eff * humid_in"),
        ("humidity_out", humidity_out, "-", "Влажность газовой фазы после ограничения насыщением"),
        ("x_h2o_after_separator", x_h2o_after_separator, "-", "Молярная доля H2O после влагоотделителя"),
        ("n_h2o_removed_mol_s", n_h2o_removed_mol_s, "mol/s", "Удалённая вода во влагоотделителе"),
        ("n_o2_accumulation_mol_s", n_o2_accumulation_mol_s, "mol/s", "Скорость накопления O2 в контуре"),
        ("n_n2_accumulation_mol_s", n_n2_accumulation_mol_s, "mol/s", "Скорость накопления N2 в контуре"),
        ("n_h2o_accumulation_mol_s", n_h2o_after_separator_mol_s, "mol/s", "Скорость накопления H2O после влагоотделителя"),
        ("n_total_impurity_accumulation_mol_s", n_total_impurity_accumulation_mol_s, "mol/s", "Суммарная скорость накопления примесей"),
        ("n_o2_accumulated_over_time_mol", n_o2_accumulation_mol_s * accumulation_time_s, "mol", "Накопление O2 за выбранный горизонт"),
        ("n_n2_accumulated_over_time_mol", n_n2_accumulation_mol_s * accumulation_time_s, "mol", "Накопление N2 за выбранный горизонт"),
        ("n_h2o_accumulated_over_time_mol", n_h2o_after_separator_mol_s * accumulation_time_s, "mol", "Накопление H2O за выбранный горизонт"),
        ("n_total_impurity_over_time_mol", n_total_impurity_accumulation_mol_s * accumulation_time_s, "mol", "Суммарное накопление примесей за выбранный горизонт"),
    ]

    return pd.DataFrame(rows, columns=["variable", "value", "unit", "meaning"])


def make_amesim_anode_state(
    temperature_K: float = ANODE_T_K,
    pressure_Pa: float = ANODE_P_PA,
    x_h2: float = ANODE_X_H2,
    x_n2: float = ANODE_X_N2,
    x_h2o: float = ANODE_X_H2O,
    x_o2: float = ANODE_X_O2,
) -> GasState:
    # Анодное состояние смеси из входных данных Amesim.
    return GasState(
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        x_h2=x_h2,
        x_n2=x_n2,
        x_h2o=x_h2o,
        x_o2=x_o2,
    )


def make_amesim_cathode_state(
    temperature_K: float = CATHODE_T_K,
    pressure_Pa: float = CATHODE_P_PA,
    x_h2: float = CATHODE_X_H2,
    x_n2: float = CATHODE_X_N2,
    x_h2o: float = CATHODE_X_H2O,
    x_o2: float = CATHODE_X_O2,
) -> GasState:
    # Катодное состояние смеси из входных данных Amesim.
    return GasState(
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        x_h2=x_h2,
        x_n2=x_n2,
        x_h2o=x_h2o,
        x_o2=x_o2,
    )


def run_demo() -> dict[str, pd.DataFrame]:
    """Выполнить демонстрационный расчёт и вернуть таблицы результата."""
    params = BTEParams()
    anode = make_amesim_anode_state()
    cathode = make_amesim_cathode_state()

    result = sc3_membrane_model(
        current_A=CURRENT_A,
        cathode=cathode,
        anode=anode,
        params=params,
    )

    return {
        "internal_variables": result_to_dataframe(result),
        "amesim_style_fluxes": sc3_amesim_flux_dataframe(result),
        "closed_recirc_accumulation": closed_recirc_impurity_accumulation_dataframe(
            result=result,
            anode=anode,
            lambda_stoich=ANODE_STOICH_LAMBDA,
            water_separator_eff_percent=WATER_SEPARATOR_EFF_PERCENT,
            accumulation_time_s=ACCUMULATION_TIME_S,
        ),
    }


def main() -> None:
    """Запустить демонстрационный расчёт из командной строки."""
    tables = run_demo()
    for title, table in tables.items():
        print(f"\n[{title}]")
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
