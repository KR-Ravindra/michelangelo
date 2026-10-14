// Copyright (c) 2023 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package oidc

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"testing"

	jose "github.com/go-jose/go-jose/v4"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/yarpc/api/transport"
	"go.uber.org/zap"

	"github.com/michelangelo-ai/michelangelo/go/auth"
)

// TestAuthenticatorSatisfiesAuthFromTokenAuthenticator proves the actual
// composition this stack relies on: a real OIDC Authenticator, verifying a
// real signed token against a real (test) issuer, adapted end to end into
// an auth.Principal via auth.FromTokenAuthenticator -- not just each piece
// tested in isolation.
func TestAuthenticatorSatisfiesAuthFromTokenAuthenticator(t *testing.T) {
	issuer := newTestIssuer(t)
	oidcAuthenticator := issuer.newAuthenticator(t, Config{})
	token := issuer.signToken(t, issuer.baseClaims())

	authenticator := auth.FromTokenAuthenticator(oidcAuthenticator)
	principal, err := authenticator.Authenticate(context.Background(), token)
	require.NoError(t, err)

	assert.Equal(t, auth.PrincipalTypeUser, principal.Type)
	assert.Equal(t, "alice@example.com", principal.Name)
	assert.Equal(t, `["dev","ops"]`, principal.Attrs["groups"])
}

// TestAuthenticatorSatisfiesAuthFromTokenAuthenticatorRejectsExpired proves
// the adapter surfaces the OIDC authenticator's own rejection (not just its
// successes) as an Authenticate error.
func TestAuthenticatorSatisfiesAuthFromTokenAuthenticatorRejectsExpired(t *testing.T) {
	issuer := newTestIssuer(t)
	oidcAuthenticator := issuer.newAuthenticator(t, Config{})
	claims := issuer.baseClaims()
	claims["exp"] = 1 // long expired
	token := issuer.signToken(t, claims)

	authenticator := auth.FromTokenAuthenticator(oidcAuthenticator)
	_, err := authenticator.Authenticate(context.Background(), token)
	assert.Error(t, err)
}

// TestAuthenticatorSatisfiesAuthFromTokenAuthenticatorRejectsWrongAudience,
// ...RejectsWrongIssuer and ...RejectsBadSignature cover the adapter's
// other negative cases end to end (not just the OIDC Authenticator in
// isolation, as oidc_test.go already does): the error propagates and no
// Principal is returned.
func TestAuthenticatorSatisfiesAuthFromTokenAuthenticatorRejectsWrongAudience(t *testing.T) {
	issuer := newTestIssuer(t)
	oidcAuthenticator := issuer.newAuthenticator(t, Config{})
	claims := issuer.baseClaims()
	claims["aud"] = "someone-else"
	token := issuer.signToken(t, claims)

	principal, err := auth.FromTokenAuthenticator(oidcAuthenticator).Authenticate(context.Background(), token)
	assert.Error(t, err)
	assert.Nil(t, principal)
}

func TestAuthenticatorSatisfiesAuthFromTokenAuthenticatorRejectsWrongIssuer(t *testing.T) {
	issuer := newTestIssuer(t)
	oidcAuthenticator := issuer.newAuthenticator(t, Config{})
	claims := issuer.baseClaims()
	claims["iss"] = "https://not-the-issuer.example.com"
	token := issuer.signToken(t, claims)

	principal, err := auth.FromTokenAuthenticator(oidcAuthenticator).Authenticate(context.Background(), token)
	assert.Error(t, err)
	assert.Nil(t, principal)
}

func TestAuthenticatorSatisfiesAuthFromTokenAuthenticatorRejectsBadSignature(t *testing.T) {
	issuer := newTestIssuer(t)
	oidcAuthenticator := issuer.newAuthenticator(t, Config{})

	// Sign with a different key under the same key ID, so only the
	// signature check can catch it -- mirrors oidc_test.go's own
	// TestAuthenticateTokenBadSignature.
	rogueKey, err := rsa.GenerateKey(rand.Reader, 2048)
	require.NoError(t, err)
	rogueSigner, err := jose.NewSigner(
		jose.SigningKey{Algorithm: jose.RS256, Key: jose.JSONWebKey{Key: rogueKey, KeyID: testKeyID}},
		nil,
	)
	require.NoError(t, err)
	payload, err := json.Marshal(issuer.baseClaims())
	require.NoError(t, err)
	jws, err := rogueSigner.Sign(payload)
	require.NoError(t, err)
	token, err := jws.CompactSerialize()
	require.NoError(t, err)

	principal, err := auth.FromTokenAuthenticator(oidcAuthenticator).Authenticate(context.Background(), token)
	assert.Error(t, err)
	assert.Nil(t, principal)
}

// TestAuthenticatorSatisfiesAuthFromTokenAuthenticatorThroughInboundMiddleware
// exercises the adapter through auth.NewInboundMiddleware (not just
// FromTokenAuthenticator.Authenticate directly), proving a real OIDC token
// actually reaches a handler as a Principal on the request context the way
// a live apiserver call would.
func TestAuthenticatorSatisfiesAuthFromTokenAuthenticatorThroughInboundMiddleware(t *testing.T) {
	issuer := newTestIssuer(t)
	oidcAuthenticator := issuer.newAuthenticator(t, Config{})
	token := issuer.signToken(t, issuer.baseClaims())

	inbound := auth.NewInboundMiddleware(auth.FromTokenAuthenticator(oidcAuthenticator), auth.InboundConfig{}, zap.NewNop())

	headers := transport.NewHeaders().With("authorization", "Bearer "+token)
	req := &transport.Request{Procedure: "Test::Call", Headers: headers}
	handler := &capturingHandler{}

	err := inbound.Handle(context.Background(), req, nil, handler)
	require.NoError(t, err)
	require.True(t, handler.called)

	principal, ok := auth.PrincipalFromContext(handler.gotCtx)
	require.True(t, ok)
	assert.Equal(t, "alice@example.com", principal.Name)
}

// capturingHandler records whether it was invoked and with what context, so
// a test can inspect the Principal the middleware stored there.
type capturingHandler struct {
	called bool
	gotCtx context.Context
}

func (h *capturingHandler) Handle(ctx context.Context, _ *transport.Request, _ transport.ResponseWriter) error {
	h.called = true
	h.gotCtx = ctx
	return nil
}
