package handler

import (
	"context"
	"errors"
	"testing"

	"github.com/go-logr/logr/funcr"
	"github.com/golang/mock/gomock"
	"github.com/michelangelo-ai/michelangelo/go/storage"
	"github.com/michelangelo-ai/michelangelo/go/storage/storagemocks"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
)

// TestHandleDeleteBlobStorageErrorLogging verifies that handleDelete logs a
// blob-storage deletion failure only when the deletion actually fails: a
// successful delete must not emit the "Failed to delete object in blob
// storage" error line.
func TestHandleDeleteBlobStorageErrorLogging(t *testing.T) {
	typeMeta := &metav1.TypeMeta{Kind: "RayJob"}

	testCases := []struct {
		name          string
		blobDeleteErr error
		wantErrorLogs int
	}{
		{
			name:          "no error log when blob delete succeeds",
			blobDeleteErr: nil,
			wantErrorLogs: 0,
		},
		{
			name:          "one error log when blob delete fails",
			blobDeleteErr: errors.New("blob backend unavailable"),
			wantErrorLogs: 1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			metadataStorage := storagemocks.NewMockMetadataStorage(ctrl)
			blobStorage := storagemocks.NewMockBlobStorage(ctrl)

			obj := &v2pb.RayJob{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "default",
					Name:      "job01",
					UID:       "uid-01",
				},
			}

			blobStorage.EXPECT().IsObjectInteresting(gomock.Any()).Return(true)
			metadataStorage.EXPECT().GetByID(gomock.Any(), "uid-01", gomock.Any()).Return(nil)
			metadataStorage.EXPECT().Delete(gomock.Any(), typeMeta, "default", "job01").Return(nil)
			blobStorage.EXPECT().DeleteFromBlobStorage(gomock.Any(), gomock.Any()).Return(tc.blobDeleteErr)

			var logLines []string
			log := funcr.New(func(prefix, args string) {
				logLines = append(logLines, args)
			}, funcr.Options{})

			err := handleDelete(context.Background(), log, typeMeta, obj, metadataStorage, NewBlobHandler(blobStorage))

			assert.NoError(t, err)
			assert.Len(t, logLines, tc.wantErrorLogs)
		})
	}
}

// TestMetadataHandlerListRemainingItemCount verifies that List copies the storage
// RemainingItemCount (set for ListOptionsExt.IncludeTotal) onto the returned list's ListMeta.
func TestMetadataHandlerListRemainingItemCount(t *testing.T) {
	require.NoError(t, v2pb.AddToScheme(scheme.Scheme))

	remaining := int64(42)
	testCases := []struct {
		name          string
		storageCount  *int64
		wantRemaining *int64
	}{
		{name: "count set when include_total requested", storageCount: &remaining, wantRemaining: &remaining},
		{name: "count unset when include_total not requested", storageCount: nil, wantRemaining: nil},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			metadataStorage := storagemocks.NewMockMetadataStorage(ctrl)
			listOptionsExt := &apipb.ListOptionsExt{IncludeTotal: tc.storageCount != nil}
			metadataStorage.EXPECT().
				List(gomock.Any(), gomock.Any(), "default", gomock.Any(), listOptionsExt, gomock.Any()).
				DoAndReturn(func(_ context.Context, _ *metav1.TypeMeta, _ string, _ *metav1.ListOptions,
					_ *apipb.ListOptionsExt, resp *storage.ListResponse) error {
					resp.Items = []runtime.Object{&v2pb.PipelineRun{ObjectMeta: metav1.ObjectMeta{Name: "run-1"}}}
					resp.Continue = "1"
					resp.RemainingItemCount = tc.storageCount
					return nil
				})

			handler := &MetadataHandlerImpl{storage: metadataStorage}
			list := &v2pb.PipelineRunList{}
			err := handler.List(context.Background(), "default", &metav1.ListOptions{}, listOptionsExt, list)

			require.NoError(t, err)
			require.Len(t, list.Items, 1)
			assert.Equal(t, "1", list.GetContinue())
			assert.Equal(t, tc.wantRemaining, list.GetRemainingItemCount())
		})
	}
}
