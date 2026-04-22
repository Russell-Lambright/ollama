package cancel

import "testing"

func TestReasonValid(t *testing.T) {
	for _, r := range []Reason{ReasonCallerCancelled, ReasonNodeFailed, ReasonOrchestratorShutdown} {
		if !r.Valid() {
			t.Errorf("Reason(%q).Valid() = false, want true", r)
		}
	}
	if Reason("bogus").Valid() {
		t.Error(`Reason("bogus").Valid() = true`)
	}
}

func TestReasonString(t *testing.T) {
	if ReasonCallerCancelled.String() != "caller_cancelled" {
		t.Errorf("ReasonCallerCancelled.String() = %q", ReasonCallerCancelled.String())
	}
}
