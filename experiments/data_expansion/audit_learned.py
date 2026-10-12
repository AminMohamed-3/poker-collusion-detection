from pathlib import Path
import audit_pilot
audit_pilot.HERE=Path(__file__).resolve().parent/'learned'
audit_pilot.main()
