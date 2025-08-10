# Create: debug_schema.py
from app.models.schemas import CustomerAnalysisResponse
import inspect

# Get the schema information
schema_info = CustomerAnalysisResponse.model_json_schema()
print("CustomerAnalysisResponse schema:")
print("=" * 50)

# Check the analysis field specifically
if 'properties' in schema_info and 'analysis' in schema_info['properties']:
    analysis_field = schema_info['properties']['analysis']
    print(f"Analysis field type: {analysis_field}")
else:
    print("Analysis field not found in schema")

# Print all fields
print("\nAll fields:")
for field_name, field_info in schema_info.get('properties', {}).items():
    field_type = field_info.get('type', 'unknown')
    print(f"  {field_name}: {field_type}")
