import pandera as pa
from pandera import Column, DataFrameSchema

data_frame_schema = DataFrameSchema({
    # Gender categories
    "gender": Column(
        str,
        nullable=False,
        checks=pa.Check.isin(['female', 'male'])
    ),
    
    # Race/ethnicity categories
    "race_ethnicity": Column(
        str,
        nullable=False,
        checks=pa.Check.isin(['group A', 'group B', 'group C', 'group D', 'group E'])
    ),
    
    # Parental education categories
    "parental_level_of_education": Column(
        str,
        nullable=False,
        checks=pa.Check.isin([
            "bachelor's degree",
            "master's degree",
            "associate's degree",
            "some college",
            "high school",
            "some high school"
        ])
    ),
    
    # Lunch categories
    "lunch": Column(
        str,
        nullable=False,
        checks=pa.Check.isin(['standard', 'free/reduced'])
    ),
    
    # Test preparation categories
    "test_preparation_course": Column(
        str,
        nullable=False,
        checks=pa.Check.isin(['none', 'completed'])
    ),
    
    # Numeric score columns
    "math_score": Column(
        int,
        nullable=False,
        checks=[
            pa.Check.greater_than_or_equal_to(0),
            pa.Check.less_than_or_equal_to(100)
        ]
    ),
    
    "reading_score": Column(
        int,
        nullable=False,
        checks=[
            pa.Check.greater_than_or_equal_to(0),
            pa.Check.less_than_or_equal_to(100)
        ]
    ),
    
    "writing_score": Column(
        int,
        nullable=False,
        checks=[
            pa.Check.greater_than_or_equal_to(0),
            pa.Check.less_than_or_equal_to(100)
        ]
    )
    })
