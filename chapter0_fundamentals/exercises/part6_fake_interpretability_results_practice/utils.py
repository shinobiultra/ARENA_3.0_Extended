def print_report(title: str, report: dict):
    print(f"\n{title}")
    for key, value in report.items():
        print(f"  {key}: {value}")
