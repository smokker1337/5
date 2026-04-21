import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# ============================
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================

def load_and_prepare(file_path):
    """Загрузка CSV, очистка и преобразование данных"""
    # Загрузка
    df = pd.read_csv(file_path, parse_dates=['date'])
    print("Исходная размерность:", df.shape)
    print("Первые 3 строки:\n", df.head(3))
    
    # Очистка: удаление пропусков
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"Удалено строк с пропусками: {before - after}")
    
    # Проверка на дубликаты
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Удалено дубликатов: {duplicates}")
    
    # Преобразование типов (убедимся, что числовые столбцы - числа)
    numeric_cols = ['revenue', 'expenses', 'customers', 'orders', 'profit']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Повторная очистка после приведения типов
    df = df.dropna()
    
    # Сортировка по дате
    df = df.sort_values('date')
    
    # Добавление полезных признаков
    df['year_month'] = df['date'].dt.to_period('M')
    df['day_of_week'] = df['date'].dt.day_name()
    
    print("\nПосле очистки:", df.shape)
    print("Информация о данных:")
    print(df.info())
    return df

# ============================
# 2. РАСЧЁТ ПОКАЗАТЕЛЕЙ
# ============================

def calculate_metrics(df):
    """Суммарные, средние, динамические и долевые показатели"""
    # Суммарные показатели
    total = df[['revenue', 'expenses', 'profit', 'customers', 'orders']].sum()
    print("\n=== СУММАРНЫЕ ПОКАЗАТЕЛИ ===")
    print(total.round(2))
    
    # Средние значения
    mean = df[['revenue', 'expenses', 'profit', 'customers', 'orders']].mean()
    print("\n=== СРЕДНИЕ ЗНАЧЕНИЯ (за день) ===")
    print(mean.round(2))
    
    # Динамика изменений (день к дню в %)
    df['revenue_change_pct'] = df['revenue'].pct_change() * 100
    df['profit_change_pct'] = df['profit'].pct_change() * 100
    
    # Долевые показатели
    df['expense_ratio'] = (df['expenses'] / df['revenue']) * 100
    df['profit_margin'] = (df['profit'] / df['revenue']) * 100
    df['conversion_rate'] = (df['orders'] / df['customers']) * 100
    
    # Средние долевые показатели
    print("\n=== ДОЛЕВЫЕ ПОКАЗАТЕЛИ (средние) ===")
    print(f"Доля расходов в выручке: {df['expense_ratio'].mean():.2f}%")
    print(f"Рентабельность (прибыль/выручка): {df['profit_margin'].mean():.2f}%")
    print(f"Конверсия (заказы/клиенты): {df['conversion_rate'].mean():.2f}%")
    
    return df, total, mean

# ============================
# 3. АНАЛИЗ: ТРЕНДЫ, СРАВНЕНИЯ, ЗАВИСИМОСТИ
# ============================

def analyze_trends_and_correlations(df):
    """Выявление трендов, сравнение периодов, поиск зависимостей"""
    # Тренд: скользящее среднее за 7 дней
    df['revenue_ma7'] = df['revenue'].rolling(window=7, min_periods=1).mean()
    df['profit_ma7'] = df['profit'].rolling(window=7, min_periods=1).mean()
    
    # Сравнение периодов: первая неделя vs последняя неделя
    if len(df) >= 14:
        first_week = df.head(7)[['revenue', 'profit', 'customers']].mean()
        last_week = df.tail(7)[['revenue', 'profit', 'customers']].mean()
        print("\n=== СРАВНЕНИЕ ПЕРВОЙ И ПОСЛЕДНЕЙ НЕДЕЛИ ===")
        compare = pd.DataFrame({'Первая неделя': first_week, 'Последняя неделя': last_week})
        compare['Изменение (%)'] = ((compare['Последняя неделя'] - compare['Первая неделя']) / compare['Первая неделя']) * 100
        print(compare.round(2))
    
    # Корреляционная матрица
    numeric_df = df[['revenue', 'expenses', 'customers', 'orders', 'profit']]
    corr = numeric_df.corr()
    print("\n=== КОРРЕЛЯЦИЯ МЕЖДУ ПОКАЗАТЕЛЯМИ ===")
    print(corr.round(3))
    
    # Сильные корреляции (выше 0.7 или ниже -0.7)
    strong_corr = corr[(corr.abs() > 0.7) & (corr != 1.0)].stack().drop_duplicates()
    if not strong_corr.empty:
        print("\nСильные корреляции:")
        for (var1, var2), val in strong_corr.items():
            print(f"  {var1} — {var2}: {val:.3f}")
    
    return df

# ============================
# 4. ВИЗУАЛИЗАЦИЯ
# ============================

def plot_all(df):
    """Построение всех необходимых графиков"""
    # 1. Линейные графики: выручка, расходы, прибыль
    plt.figure(figsize=(14, 6))
    plt.plot(df['date'], df['revenue'], label='Выручка', marker='o', linewidth=2)
    plt.plot(df['date'], df['expenses'], label='Расходы', marker='s', linewidth=2)
    plt.plot(df['date'], df['profit'], label='Прибыль', marker='^', linewidth=2)
    plt.title('Динамика выручки, расходов и прибыли', fontsize=14)
    plt.xlabel('Дата')
    plt.ylabel('Сумма')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('line_dynamics.png', dpi=150)
    plt.show()
    
    # 2. Столбчатая диаграмма: сравнение средних показателей по дням недели
    weekday_avg = df.groupby('day_of_week')[['revenue', 'profit']].mean().reindex(
        ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    )
    weekday_avg.plot(kind='bar', figsize=(12, 5), rot=45)
    plt.title('Средняя выручка и прибыль по дням недели')
    plt.ylabel('Средняя сумма')
    plt.tight_layout()
    plt.savefig('bar_weekday.png', dpi=150)
    plt.show()
    
    # 3. Круговая диаграмма: доля расходов и прибыли в общей выручке
    total_rev = df['revenue'].sum()
    total_exp = df['expenses'].sum()
    total_profit = df['profit'].sum()
    labels = ['Расходы', 'Прибыль']
    sizes = [total_exp, total_profit]
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, explode=(0.05, 0))
    plt.title(f'Структура выручки (всего {total_rev:,.0f} руб.)')
    plt.savefig('pie_structure.png', dpi=150)
    plt.show()
    
    # 4. Тепловая карта корреляций
    plt.figure(figsize=(8, 6))
    corr = df[['revenue', 'expenses', 'customers', 'orders', 'profit']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=1)
    plt.title('Корреляционная матрица показателей')
    plt.tight_layout()
    plt.savefig('heatmap_corr.png', dpi=150)
    plt.show()
    
    # 5. Сравнительный отчёт: выручка и прибыль со скользящим средним
    plt.figure(figsize=(14, 5))
    plt.plot(df['date'], df['revenue'], alpha=0.5, label='Выручка (факт)')
    plt.plot(df['date'], df['revenue_ma7'], 'r--', linewidth=2, label='Тренд выручки (MA7)')
    plt.plot(df['date'], df['profit'], alpha=0.5, label='Прибыль (факт)')
    plt.plot(df['date'], df['profit_ma7'], 'g--', linewidth=2, label='Тренд прибыли (MA7)')
    plt.title('Фактические данные и тренды (скользящее среднее за 7 дней)')
    plt.xlabel('Дата')
    plt.ylabel('Сумма')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('trend_comparison.png', dpi=150)
    plt.show()

# ============================
# 5. ФОРМИРОВАНИЕ АНАЛИТИЧЕСКОГО ОТЧЁТА
# ============================

def generate_report(df, total, mean):
    """Вывод текстового отчёта с выводами"""
    report = []
    report.append("\n" + "="*60)
    report.append("АНАЛИТИЧЕСКИЙ ОТЧЁТ ПО ДЕЯТЕЛЬНОСТИ ОРГАНИЗАЦИИ")
    report.append("="*60)
    
    # Описание данных
    report.append("\n1. ОПИСАНИЕ ДАННЫХ")
    report.append(f"   Период: с {df['date'].min().date()} по {df['date'].max().date()}")
    report.append(f"   Количество дней: {len(df)}")
    report.append("   Показатели: выручка, расходы, число клиентов, число заказов, прибыль")
    
    # Расчёт показателей
    report.append("\n2. КЛЮЧЕВЫЕ ПОКАЗАТЕЛИ")
    report.append(f"   Общая выручка: {total['revenue']:,.2f} руб.")
    report.append(f"   Общие расходы: {total['expenses']:,.2f} руб.")
    report.append(f"   Общая прибыль: {total['profit']:,.2f} руб.")
    report.append(f"   Средняя дневная выручка: {mean['revenue']:,.2f} руб.")
    report.append(f"   Средняя дневная прибыль: {mean['profit']:,.2f} руб.")
    report.append(f"   Средняя маржинальность: {df['profit_margin'].mean():.2f}%")
    report.append(f"   Средняя конверсия (заказы/клиенты): {df['conversion_rate'].mean():.2f}%")
    
    # Аналитические выводы
    report.append("\n3. АНАЛИТИЧЕСКИЕ ВЫВОДЫ")
    
    # Тренд
    if len(df) > 1:
        revenue_trend = df['revenue'].iloc[-1] - df['revenue'].iloc[0]
        profit_trend = df['profit'].iloc[-1] - df['profit'].iloc[0]
        if revenue_trend > 0:
            report.append(f"   • Выручка за период выросла на {revenue_trend:,.2f} руб. (положительный тренд)")
        else:
            report.append(f"   • Выручка снизилась на {abs(revenue_trend):,.2f} руб. (отрицательный тренд)")
        if profit_trend > 0:
            report.append(f"   • Прибыль увеличилась на {profit_trend:,.2f} руб.")
        else:
            report.append(f"   • Прибыль уменьшилась на {abs(profit_trend):,.2f} руб.")
    
    # Корреляции
    corr_rev_profit = df['revenue'].corr(df['profit'])
    corr_cust_orders = df['customers'].corr(df['orders'])
    report.append(f"   • Корреляция выручки и прибыли: {corr_rev_profit:.3f} (сильная связь)")
    report.append(f"   • Корреляция числа клиентов и заказов: {corr_cust_orders:.3f} (ожидаемо высокая)")
    
    # Долевые рекомендации
    exp_ratio = df['expense_ratio'].mean()
    if exp_ratio > 80:
        report.append(f"   • Высокая доля расходов ({exp_ratio:.1f}%) – рекомендуется оптимизация затрат.")
    else:
        report.append(f"   • Доля расходов ({exp_ratio:.1f}%) находится в приемлемых пределах.")
    
    # Сезонность (если есть данные по дням недели)
    best_day = df.groupby('day_of_week')['revenue'].mean().idxmax()
    report.append(f"   • Максимальная выручка в среднем приходится на {best_day}.")
    
    report.append("\n" + "="*60)
    report.append("Графики сохранены в файлы: line_dynamics.png, bar_weekday.png, pie_structure.png, heatmap_corr.png, trend_comparison.png")
    report.append("="*60)
    
    # Печать отчёта
    for line in report:
        print(line)
    
    # Сохранение отчёта в текстовый файл
    with open('analytical_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

# ============================
# ГЛАВНАЯ ФУНКЦИЯ (ЗАПУСК ВСЕГО ПРОЕКТА)
# ============================

def main():
    # Укажите путь к файлу (если файл лежит в той же папке, достаточно имени)
    file_path = "PP12_ISP22_analytics.csv"
    
    # Загрузка и подготовка
    df = load_and_prepare(file_path)
    
    # Расчёт показателей
    df, total, mean = calculate_metrics(df)
    
    # Анализ трендов и зависимостей
    df = analyze_trends_and_correlations(df)
    
    # Визуализация
    plot_all(df)
    
    # Формирование отчёта
    generate_report(df, total, mean)
    
    print("\nПроект успешно выполнен. Результаты сохранены: графики (PNG) и отчёт (analytical_report.txt).")

# Запуск
if __name__ == "__main__":
    main()