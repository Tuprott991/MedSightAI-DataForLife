import { KeyFindings } from './KeyFindings';
import { SuspectedDisease } from './SuspectedDisease';

export const AnalysisTab = ({ findings, suspectedDiseases, onFindingClick, onFindingSelectionChange, onUpdateClick, isUpdating, selectedFindingIds }) => {
    return (
        <div className="space-y-4">
            <SuspectedDisease
                diseases={suspectedDiseases || []}
                onUpdateClick={onUpdateClick}
                isUpdating={isUpdating}
                selectedFindingIds={selectedFindingIds}
            />
            <KeyFindings
                findings={findings}
                onFindingClick={onFindingClick}
                onFindingSelectionChange={onFindingSelectionChange}
            />
        </div>
    );
};
