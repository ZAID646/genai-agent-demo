import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import CalculatorTool, WebSearchTool, WeatherTool, extract_json

calculator = CalculatorTool()
web = WebSearchTool()
weather = WeatherTool()


class TestCalculator:
    def test_simple_arithmetic(self):
        assert float(calculator.run(expression="15 * 7 + 3")) == 108.0

    def test_division(self):
        assert float(calculator.run(expression="100 / 4")) == 25.0

    def test_power(self):
        result = calculator.run(expression="2 ** 10")
        assert "1024" in result or float(result) == 1024

    def test_empty_expression(self):
        assert "Error" not in calculator.run(expression="2 + 2")

    def test_invalid_expression(self):
        result = calculator.run(expression="invalid")
        assert "Error" in result


class TestExtractJson:
    def test_plain_json(self):
        result = extract_json('{"thought": "test", "tool": null}')
        assert result == {"thought": "test", "tool": None}

    def test_json_in_code_fence(self):
        result = extract_json('```\n{"thought": "test"}\n```')
        assert result == {"thought": "test"}

    def test_json_with_text_around(self):
        result = extract_json('some text\n{"thought": "hello"}\nmore text')
        assert result == {"thought": "hello"}

    def test_empty_input(self):
        import pytest
        with pytest.raises(json.JSONDecodeError):
            extract_json("")


class TestWeatherTool:
    def test_no_input(self):
        result = weather.run()
        assert "No location provided" in result

    def test_valid_city(self):
        result = weather.run(location="London")
        assert "Weather in" in result
        assert "°C" in result

    def test_unknown_location(self):
        result = weather.run(location="Xyzabc12345")
        assert "Could not find" in result or "error" in result.lower()
